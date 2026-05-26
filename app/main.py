from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from llama_stack_client import LlamaStackClient
from openai import OpenAI
from mlflow.entities import AssessmentSource, AssessmentSourceType
import os
import asyncio
import json
import logging
import threading
import queue
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import random
import mlflow

app = FastAPI(title="Canopy Backend API")

# MLflow prompt registry configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.redhat-ods-applications.svc.cluster.local:8443")
if not os.getenv("MLFLOW_TRACKING_AUTH"):
    os.environ["MLFLOW_TRACKING_AUTH"] = "kubernetes"
if not os.getenv("MLFLOW_TRACKING_INSECURE_TLS"):
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

NAMESPACE_PATH = "/run/secrets/kubernetes.io/serviceaccount/namespace"
APP_NAMESPACE = os.getenv("MLFLOW_WORKSPACE")
TOOLINGS_NAMESPACE = os.getenv("MLFLOW_PROMPT_WORKSPACE")

if not APP_NAMESPACE and os.path.exists(NAMESPACE_PATH):
    with open(NAMESPACE_PATH) as f:
        APP_NAMESPACE = f.read().strip()

if APP_NAMESPACE:
    os.environ["MLFLOW_WORKSPACE"] = APP_NAMESPACE
    if not TOOLINGS_NAMESPACE:
        if APP_NAMESPACE.endswith(("-test", "-prod")):
            user_prefix = APP_NAMESPACE.rsplit("-", 1)[0]
            TOOLINGS_NAMESPACE = f"{user_prefix}-toolings"
        else:
            TOOLINGS_NAMESPACE = APP_NAMESPACE

SA_TOKEN_PATH = "/run/secrets/kubernetes.io/serviceaccount/token"
if os.path.exists(SA_TOKEN_PATH):
    with open(SA_TOKEN_PATH) as f:
        os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.openai.autolog()
mlflow.langchain.autolog()
logging.getLogger("mlflow.langchain.langchain_tracer").setLevel(logging.ERROR)

# =============================================================================
# Refusal Signature Detection
#
# NeMo Guardrails returns the bot's refusal message as normal response content
# when a rail blocks. Each bot message in rails.co starts with a unique emoji,
# so we detect the guardrail by checking the first content chunk.
#
# Maps: emoji_prefix → (detector_name, frontend_css_class)
# =============================================================================

REFUSAL_SIGNATURES: dict[str, tuple[str, str | None]] = {
    "🌐": ("language", "language"),
    "🚫": ("hap_or_regex", "hap"),
    "⚠️": ("regex", "regex"),
    "🛡": ("prompt_injection", "prompt-injection"),  # 🛡️ starts with 🛡 (U+1F6E1)
    "😔": ("hap", "hap"),
    "🤔": ("topic_relevance", None),
    "🔒": ("pii", None),
}


def _detect_refusal(content: str) -> tuple[str | None, str | None]:
    """Return (detector_name, css_class) if content is a refusal, else (None, None)."""
    stripped = content.lstrip()
    for emoji, info in REFUSAL_SIGNATURES.items():
        if stripped.startswith(emoji):
            return info
    return None, None


def get_mlflow_prompt(feature, version_override=None):
    """Fetch system prompt from MLflow prompt registry for a given feature.

    Temporarily switches to the toolings workspace to load prompts,
    then switches back to the app's own workspace for experiment tracking.
    """
    prompt_name = config[feature]["mlflow_prompt"]
    version = version_override or config[feature].get("mlflow_prompt_version", "latest")
    original_workspace = os.environ.get("MLFLOW_WORKSPACE")
    try:
        if TOOLINGS_NAMESPACE:
            os.environ["MLFLOW_WORKSPACE"] = TOOLINGS_NAMESPACE
        if version.isdigit():
            prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}/{version}")
        else:
            prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@{version}")
    finally:
        if original_workspace:
            os.environ["MLFLOW_WORKSPACE"] = original_workspace
    return prompt.template

# Load configuration with fallback for testing
config_path = os.getenv("CANOPY_CONFIG_PATH", "/canopy/canopy-config.yaml")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    _openai_clients = {}
    def get_openai_client(feature):
        endpoint = config[feature]["endpoint"]
        if endpoint not in _openai_clients:
            base_url = endpoint if endpoint.endswith('/') else endpoint + '/'
            _openai_clients[endpoint] = OpenAI(base_url=base_url, api_key="no-key-required")
        return _openai_clients[endpoint]

    _llama_clients = {}
    def get_llama_client(feature):
        endpoint = config[feature]["endpoint"]
        if endpoint not in _llama_clients:
            # LlamaStackClient adds /v1/ internally; strip it to avoid /v1/v1/ doubling
            llama_base = endpoint.rstrip('/').removesuffix('/v1')
            _llama_clients[endpoint] = LlamaStackClient(base_url=llama_base)
        return _llama_clients[endpoint]

    # Feature flags configuration from environment variables
    FEATURE_FLAGS = {
        "information-search": "information-search" in config and config["information-search"]["enabled"] == True,
        "summarization": "summarization" in config and config["summarization"]["enabled"] == True,
        "student-assistant": "student-assistant" in config and config["student-assistant"]["enabled"] == True,
        "shields": "shields" in config and config["shields"]["enabled"] == True,
        "feedback": "feedback" in config and config["feedback"]["enabled"] == True,
        "ab_testing": "ab_testing" in config and config["ab_testing"]["enabled"] == True,
        "socratic-tutor": "socratic-tutor" in config and config["socratic-tutor"]["enabled"] == True,
    }

    # Shields configuration
    SHIELDS_CONFIG = {}
else:
    # Fallback for testing - config will be loaded by tests
    config = None
    def get_openai_client(feature): return None
    def get_llama_client(feature): return None
    FEATURE_FLAGS = {
        "information-search": False,
        "summarization": False,
        "student-assistant": False,
        "shields": False,
        "feedback": False,
        "ab_testing": False,
        "socratic-tutor": False,
    }
    SHIELDS_CONFIG = {}

# Tool factory function for creating student assistant tools
# This allows tools to be imported and tested independently
def create_student_tools(llama_stack_client: LlamaStackClient, vector_store_id: str):
    """
    Factory function to create student assistant tools.

    This function is used both by the agent in production and by tests.
    It ensures a single source of truth for tool implementations.

    Args:
        llama_stack_client: LlamaStackClient instance for vector search
        vector_store_id: ID of the vector store to search

    Returns:
        List of LangChain tools (search_knowledge_base, find_professors_by_expertise)
    """
    # Professor directory - used by find_professors_by_expertise tool
    PROFESSORS = {
        "Dr. Sarah Chen": {
            "department": "Computer Science",
            "expertise": ["Machine Learning", "Neural Networks", "AI Ethics", "Agentic Workflows"],
            "email": "s.chen@university.edu"
        },
        "Prof. Michael Rodriguez": {
            "department": "Physics",
            "expertise": ["Quantum Mechanics", "Particle Physics", "Quantum Chromodynamics"],
            "email": "m.rodriguez@university.edu"
        },
        "Dr. Emily Thompson": {
            "department": "Biology",
            "expertise": ["Botany", "Ecology", "Forest Canopy Structure", "Plant Biology"],
            "email": "e.thompson@university.edu"
        },
        "Prof. James Wilson": {
            "department": "Computer Science",
            "expertise": ["Distributed Systems", "Cloud Computing", "Software Architecture"],
            "email": "j.wilson@university.edu"
        }
    }

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search through documents to find information. Use this when the user asks about concepts, definitions, or topics."""
        try:
            results = llama_stack_client.vector_stores.search(
                vector_store_id=vector_store_id,
                query=query,
                max_num_results=3,
                search_mode="vector",
                ranking_options={"score_threshold": 0.0}
            )
            if not results.data:
                return "No relevant information found in the knowledge base."
            formatted_results = []
            for i, result in enumerate(results.data, 1):
                content = result.content[0].text if hasattr(result, 'content') else str(result)
                formatted_results.append(f"Result {i}: {content}")
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    @tool
    def find_professors_by_expertise(topic: str) -> str:
        """Find professors who have expertise in a specific topic or subject area."""
        matching_profs = []
        for name, info in PROFESSORS.items():
            if any(topic.lower() in exp.lower() or exp.lower() in topic.lower() for exp in info["expertise"]):
                matching_profs.append((name, info))

        if not matching_profs:
            result = f"No professors found with specific expertise in '{topic}'.\n\nAvailable professors:\n\n"
            for name, info in PROFESSORS.items():
                result += f"**{name}** - {info['department']}\n  Expertise: {', '.join(info['expertise'])}\n  Email: {info['email']}\n\n"
            return result

        result = f"Professors with expertise in '{topic}':\n\n"
        for name, info in matching_profs:
            result += f"**{name}** - {info['department']}\n  Expertise: {', '.join(info['expertise'])}\n  Email: {info['email']}\n\n"
        return result

    return [search_knowledge_base, find_professors_by_expertise]


# Initialize student assistant agent if enabled
agent = None
if FEATURE_FLAGS.get("student-assistant", False):
    from datetime import datetime
    vector_store_id = config["student-assistant"].get("vector_db_id", "latest")

    # Create tools using factory function
    student_tools = create_student_tools(get_llama_client("student-assistant"), vector_store_id)

    tools = student_tools + [
        {
            "type": "mcp",
            "server_label": "canopy-calendar",
            "server_url": config["student-assistant"].get("mcp_calendar_url", "http://canopy-mcp-calendar-mcp-server:8080/sse"),
            "require_approval": "never",
        }
    ]

    llm = ChatOpenAI(
        openai_api_base=config["student-assistant"]["endpoint"],
        model=config["student-assistant"]["model"],
        openai_api_key="not-needed",
        use_responses_api=True,
        temperature=config["student-assistant"].get("temperature", 0.1)
    )

    # Evaluate the prompt as an f-string to execute any Python expressions in {}
    prompt_template = get_mlflow_prompt("student-assistant")
    formatted_prompt = eval(f'f"""{prompt_template}"""')

    agent = create_react_agent(
        llm,
        tools,
        prompt=formatted_prompt,
        checkpointer=MemorySaver()
    )

class PromptRequest(BaseModel):
    prompt: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    trace_id: str
    rating: str  # "thumbs_up" or "thumbs_down"
    feature: str = "summarization"
    comment: Optional[str] = None

class ABFeedbackRequest(BaseModel):
    trace_id_a: str
    trace_id_b: str
    preference: str  # "a" or "b"
    prompt_mapping: Dict[str, str]  # {"a": "prompt", "b": "prompt_b"}
    feature: str = "summarization"

@app.get("/feature-flags")
async def get_feature_flags() -> Dict[str, Any]:
    """Get all feature flags configuration"""
    return FEATURE_FLAGS

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Attach thumbs up/down feedback to the MLflow trace for this response."""
    if not FEATURE_FLAGS.get("feedback", False):
        raise HTTPException(status_code=404, detail="Feedback feature is not enabled")

    for attempt in range(8):
        try:
            mlflow.log_feedback(
                trace_id=request.trace_id,
                name="user_satisfaction",
                value=(request.rating == "thumbs_up"),
                rationale=request.comment,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id="canopy_user",
                ),
            )
            return {"status": "ok"}
        except Exception as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e) and attempt < 7:
                await asyncio.sleep(1)
            else:
                raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarization/ab")
async def summarization_ab(request: PromptRequest, raw_request: Request):
    """Run the same input through two different prompts for A/B comparison."""
    if not FEATURE_FLAGS.get("summarization", False):
        raise HTTPException(status_code=404, detail="Summarization feature is not enabled")
    if not FEATURE_FLAGS.get("ab_testing", False):
        raise HTTPException(status_code=404, detail="A/B testing feature is not enabled")

    session_id = raw_request.headers.get("x-session-id")
    prompt_b_version = config.get("ab_testing", {}).get("mlflow_prompt_b_version", "latest")
    prompt_a_text = get_mlflow_prompt("summarization")
    prompt_b_text = get_mlflow_prompt("summarization", version_override=prompt_b_version)

    temperature = config["summarization"].get("temperature", 0.7)
    max_tokens = config["summarization"].get("max_tokens", 4096)
    model = config["summarization"]["model"]

    # Randomize which prompt is A vs B to avoid position bias
    prompts = [("champion", prompt_a_text), ("challenger", prompt_b_text)]
    random.shuffle(prompts)
    mapping = {"a": prompts[0][0], "b": prompts[1][0]}

    q = queue.Queue()

    mlflow.set_experiment("summarization")

    def ab_worker(variant, sys_prompt):
        """Run inference for one variant, tag chunks, and emit trace_id after flush."""
        ab_trace_id_holder = []

        @mlflow.trace(name=f"summarization_ab_{variant}")
        def run(messages: list) -> str:
            trace_id = mlflow.get_active_trace_id()
            if trace_id:
                ab_trace_id_holder.append(trace_id)
            full_response = ""
            response = get_openai_client("summarization").chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for r in response:
                if hasattr(r, 'choices') and r.choices:
                    delta = r.choices[0].delta
                    if delta.content:
                        full_response += delta.content
                        chunk = f"data: {json.dumps({'variant': variant, 'delta': delta.content})}\n\n"
                        q.put(chunk)
            return full_response

        ab_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": request.prompt},
        ]
        try:
            run(ab_messages)
            if ab_trace_id_holder:
                q.put(f"data: {json.dumps({'type': f'trace_id_{variant}', 'trace_id': ab_trace_id_holder[0]})}\n\n")
        except Exception as e:
            q.put(f"data: {json.dumps({'variant': variant, 'error': str(e)})}\n\n")
        finally:
            q.put(("DONE", variant))

    # Send config event first, then start workers
    threading.Thread(target=ab_worker, args=("a", prompts[0][1])).start()
    threading.Thread(target=ab_worker, args=("b", prompts[1][1])).start()

    async def streamer():
        # First event: send the mapping (frontend stores but doesn't show)
        yield f"data: {json.dumps({'type': 'ab_config', 'mapping': mapping})}\n\n"
        done_count = 0
        while done_count < 2:
            item = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if isinstance(item, tuple) and item[0] == "DONE":
                done_count += 1
            else:
                yield item
        yield "data: [DONE]\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")


@app.post("/feedback/ab")
async def submit_ab_feedback(request: ABFeedbackRequest):
    """Attach A/B preference as feedback on both MLflow traces."""
    if not FEATURE_FLAGS.get("feedback", False):
        raise HTTPException(status_code=404, detail="Feedback feature is not enabled")
    if not FEATURE_FLAGS.get("ab_testing", False):
        raise HTTPException(status_code=404, detail="A/B testing feature is not enabled")

    winning_prompt = request.prompt_mapping.get(request.preference, "unknown")
    winning_trace_id = request.trace_id_a if request.preference == "a" else request.trace_id_b
    losing_trace_id = request.trace_id_b if request.preference == "a" else request.trace_id_a

    for attempt in range(8):
        try:
            mlflow.log_feedback(
                trace_id=winning_trace_id,
                name="ab_preference",
                value=True,
                rationale=f"User preferred this response (mapped to {winning_prompt})",
                source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="canopy_user"),
            )
            mlflow.log_feedback(
                trace_id=losing_trace_id,
                name="ab_preference",
                value=False,
                rationale=f"User preferred the alternative response (mapped to {winning_prompt})",
                source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="canopy_user"),
            )
            return {"status": "ok"}
        except Exception as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e) and attempt < 7:
                await asyncio.sleep(1)
            else:
                raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarization")
async def summarization(request: PromptRequest, raw_request: Request):
    if not FEATURE_FLAGS.get("summarization", False):
        raise HTTPException(status_code=404, detail="Summarization feature is not enabled")

    sys_prompt = get_mlflow_prompt("summarization")
    temperature = config["summarization"].get("temperature", 0.7)
    max_tokens = config["summarization"].get("max_tokens", 4096)
    model = config["summarization"]["model"]
    session_id = raw_request.headers.get("x-session-id")

    q = queue.Queue()

    mlflow.set_experiment("summarization")

    @mlflow.trace
    def worker(messages: list[dict], session_id: str):
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id},
        )
        print(f"sending request to model {model} (direct model endpoint)")
        full_response = ""
        try:
            response = get_openai_client("summarization").chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_response += delta.content
                        q.put(f"data: {json.dumps({'delta': delta.content})}\n\n")
        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        finally:
            q.put(None)
        return full_response

    summarize_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": request.prompt},
    ]
    threading.Thread(target=worker, args=(summarize_messages, session_id)).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

@app.post("/summarization/chat")
async def summarization_chat(request: ChatRequest, raw_request: Request):
    """Chat endpoint for summarization with conversation history."""
    if not FEATURE_FLAGS.get("summarization", False):
        raise HTTPException(status_code=404, detail="Summarization feature is not enabled")

    sys_prompt = get_mlflow_prompt("summarization")
    temperature = config["summarization"].get("temperature", 0.7)
    max_tokens = config["summarization"].get("max_tokens", 4096)
    model = config["summarization"]["model"]
    session_id = raw_request.headers.get("x-session-id") or request.session_id

    # Convert ChatMessage objects to dict format for the LLM
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    q = queue.Queue()

    shields_full_messages = [{"role": "system", "content": sys_prompt}] + messages
    current_user_message = messages[-1]["content"] if messages else ""
    trace_id_holder = []

    @mlflow.trace
    def worker_with_shields(messages: list[dict], user_message: str, session_id: str):
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            trace_id_holder.append(trace_id)
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
        shield_model = config["summarization"]["model"] 

        first_content_seen = False
        is_refusal = False
        refusal_buffer = ""
        full_response = ""

        try:
            nemo_client = get_openai_client("shields")
            stream = nemo_client.chat.completions.create(
                model=shield_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                if not (hasattr(chunk, "choices") and chunk.choices):
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue

                # Peek at the first chunk to detect a refusal emoji prefix.
                if not first_content_seen:
                    first_content_seen = True
                    detector, css_class = _detect_refusal(content)
                    is_refusal = detector is not None

                if is_refusal:
                    refusal_buffer += content
                else:
                    full_response += content
                    q.put(f"data: {json.dumps({'delta': content})}\n\n")

            if is_refusal:
                full_response = refusal_buffer.strip()
                q.put(f"data: {json.dumps({'delta': full_response})}\n\n")

        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        return full_response

    mlflow.set_experiment("summarization")

    @mlflow.trace
    def worker_without_shields(messages: list[dict], session_id: str):
        """Use direct inference API when shields are disabled."""
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id},
        )
        trace_id = mlflow.get_active_trace_id()
        if trace_id:
            trace_id_holder.append(trace_id)
        print(f"sending chat request to model {model} without shields (Inference API)")
        full_response = ""
        try:
            response = get_openai_client("summarization").chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_response += delta.content
                        q.put(f"data: {json.dumps({'delta': delta.content})}\n\n")
        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        return full_response

    def run_without_shields():
        worker_without_shields(full_messages, session_id)
        if trace_id_holder:
            q.put(f"data: {json.dumps({'type': 'trace_id', 'trace_id': trace_id_holder[0]})}\n\n")
        q.put(None)

    def run_with_shields():
        worker_with_shields(shields_full_messages, current_user_message, session_id)
        if trace_id_holder:
            q.put(f"data: {json.dumps({'type': 'trace_id', 'trace_id': trace_id_holder[0]})}\n\n")
        q.put(None)

    # Choose worker based on shields feature flag
    if FEATURE_FLAGS.get("shields", False):
        threading.Thread(target=run_with_shields).start()
    else:
        full_messages = [{"role": "system", "content": sys_prompt}] + messages
        threading.Thread(target=run_without_shields).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

@app.post("/socratic-tutor")
async def socratic_tutor(request: PromptRequest, raw_request: Request):
    # Check if socratic tutor feature is enabled
    if not FEATURE_FLAGS.get("socratic-tutor", False):
        raise HTTPException(status_code=404, detail="Socratic tutor feature is not enabled")

    sys_prompt = get_mlflow_prompt("socratic-tutor")
    temperature = config["socratic-tutor"].get("temperature", 0.9)
    max_tokens = config["socratic-tutor"].get("max_tokens", 1500)
    model = config["socratic-tutor"]["model"]
    session_id = raw_request.headers.get("x-session-id")

    q = queue.Queue()

    mlflow.set_experiment("socratic-tutor")

    @mlflow.trace
    def worker(messages: list[dict], session_id: str):
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id},
        )
        full_response = ""
        try:
            response = get_openai_client("socratic-tutor").chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            for r in response:
                if hasattr(r, 'choices') and r.choices:
                    delta = r.choices[0].delta
                    if delta.content:
                        full_response += delta.content
                        chunk = f"data: {json.dumps({'delta': delta.content})}\n\n"
                        q.put(chunk)

        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        finally:
            q.put(None)
        return full_response

    socratic_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": request.prompt},
    ]
    threading.Thread(target=worker, args=(socratic_messages, session_id)).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

@app.post("/information-search")
async def information_search(request: PromptRequest, raw_request: Request):
    # Check if information search feature is enabled
    if not FEATURE_FLAGS.get("information-search", False):
        raise HTTPException(status_code=404, detail="Information search feature is not enabled")

    sys_prompt = get_mlflow_prompt("information-search")
    temperature = config["information-search"].get("temperature", 0.7)
    max_tokens = config["information-search"].get("max_tokens", 4096)
    vector_db_id = config["information-search"].get("vector_db_id", "latest")
    session_id = raw_request.headers.get("x-session-id")

    q = queue.Queue()

    mlflow.set_experiment("information-search")

    @mlflow.trace(span_type="RETRIEVER")
    def retrieve_chunks(query: str) -> list[str]:
        results = get_llama_client("information-search").vector_stores.search(
            vector_store_id=vector_db_id,
            query=query,
            max_num_results=5,
            search_mode="vector"
        )
        return [
            result.content[0].text if hasattr(result, 'content') else str(result)
            for result in results.data
        ]

    def worker(query: str, session_id: str):
        with mlflow.start_span(name=query, span_type="CHAIN") as span:
            trace_id = getattr(span, 'request_id', None) or mlflow.get_active_trace_id()
            print(f"[information-search] trace_id={trace_id}")
            span.set_inputs(query)

            print(f"Searching in collection {vector_db_id}")
            print(f"Existing collections: {get_llama_client('information-search').vector_stores.list()}")
            print(f"query: {query}")

            chunks = retrieve_chunks(query)
            span.set_attributes({
                "mlflow.trace.session": session_id,
                "vector_store_id": vector_db_id,
                "retrieved_chunks": str(chunks),
                "num_chunks_retrieved": str(len(chunks)),
            })
            prompt_context = "\n\n".join(chunks)

            enhanced_prompt = f"""Please answer the given query using the document intelligence context below.

    CONTEXT (Processed with Docling Document Intelligence):
    {prompt_context}

    QUERY:
    {query}

    Note: The context includes intelligently processed content with preserved tables, formulas, figures, and document structure."""

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": enhanced_prompt},
            ]

            full_response = ""
            try:
                response = get_openai_client("information-search").chat.completions.create(
                    model=config["information-search"]["model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                for r in response:
                    if hasattr(r, 'choices') and r.choices:
                        delta = r.choices[0].delta
                        if delta.content:
                            full_response += delta.content
                            chunk = f"data: {json.dumps({'delta': delta.content})}\n\n"
                            q.put(chunk)
            except Exception as e:
                q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
            finally:
                if trace_id:
                    q.put(f"data: {json.dumps({'type': 'trace_id', 'trace_id': trace_id})}\n\n")
                q.put(None)
            span.set_outputs({"response": full_response})

    threading.Thread(target=worker, args=(request.prompt, session_id)).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

@app.post("/student-assistant")
async def student_assistant_chat(request: PromptRequest, raw_request: Request):
    if not FEATURE_FLAGS.get("student-assistant", False):
        raise HTTPException(status_code=404, detail="Student assistant feature is not enabled")

    if not agent:
        raise HTTPException(status_code=500, detail="Student assistant not initialized")

    session_id = raw_request.headers.get("x-session-id")
    q = queue.Queue()

    mlflow.set_experiment("student-assistant")

    def worker():
        try:
            with mlflow.start_span(name=request.prompt, span_type="AGENT") as span:
                trace_id = getattr(span, 'request_id', None) or mlflow.get_active_trace_id()
                span.set_inputs(request.prompt)

                thread_id = session_id or str(int(random.random() * 1000000))
                config_params = {"configurable": {"thread_id": thread_id}}
                inputs = {"messages": [{"role": "user", "content": request.prompt}]}

                mlflow.update_current_trace(
                    metadata={"mlflow.trace.session": thread_id},
                )

                result = agent.invoke(inputs, config_params)
                messages = result.get("messages", [])

                # Send intermediate steps (tool calls and results) to frontend
                for msg in messages:
                    msg_type = getattr(msg, "type", None)

                    if msg_type == "ai" and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            tool_call_data = {
                                "type": "tool_call",
                                "name": tc.get("name"),
                                "args": tc.get("args")
                            }
                            chunk = f"data: {json.dumps(tool_call_data)}\n\n"
                            q.put(chunk)

                    elif msg_type == "tool":
                        tool_result_data = {
                            "type": "tool_result",
                            "name": msg.name,
                            "content": str(msg.content)
                        }
                        chunk = f"data: {json.dumps(tool_result_data)}\n\n"
                        q.put(chunk)

                # Extract and send final answer
                final_answer = ""
                if messages:
                    for msg in reversed(messages):
                        if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
                            content = msg.content
                            if isinstance(content, list):
                                text_parts = []
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        text_parts.append(part.get("text", ""))
                                content = "".join(text_parts)

                            if content:
                                final_answer = content
                                chunk = f"data: {json.dumps({'type': 'final_answer'})}\n\n"
                                q.put(chunk)
                                for char in content:
                                    chunk = f"data: {json.dumps({'delta': char})}\n\n"
                                    q.put(chunk)
                            break

                span.set_outputs({"response": final_answer})

                if trace_id:
                    q.put(f"data: {json.dumps({'type': 'trace_id', 'trace_id': trace_id})}\n\n")
        except Exception as e:
            q.put(f"data: {json.dumps({'error': str(e)})}\n\n")
        finally:
            q.put(None)

    threading.Thread(target=worker).start()

    async def streamer():
        while True:
            chunk = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(streamer(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)