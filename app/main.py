from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from llama_stack_client import LlamaStackClient
import os
import asyncio
from fastapi.responses import StreamingResponse
import json
import threading
import queue
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import random

app = FastAPI(title="Canopy Backend API")

config_path = "/canopy/canopy-config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

llama_client = LlamaStackClient(base_url=config["LLAMA_STACK_URL"])

# Feature flags configuration from environment variables
FEATURE_FLAGS = {
    "information-search": "information-search" in config and config["information-search"]["enabled"] == True,
    "summarize": "summarize" in config and config["summarize"]["enabled"] == True,
    "student-assistant": "student-assistant" in config and config["student-assistant"]["enabled"] == True,
    "shields": "shields" in config and config["shields"]["enabled"] == True,
}

# Shields configuration
SHIELDS_CONFIG = {}
if FEATURE_FLAGS.get("shields", False):
    SHIELDS_CONFIG = {
        "input_shields": config["shields"].get("input_shields", []),
        "output_shields": config["shields"].get("output_shields", []),
        "model": config["shields"].get("model", config.get("student-assistant", {}).get("model", "llama32")),
    }

# Professor directory
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

# Initialize student assistant agent if enabled
agent = None
if FEATURE_FLAGS.get("student-assistant", False):
    vector_store_id = config["student-assistant"].get("vector_db_id", "latest")

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search through documents to find information. Use this when the user asks about concepts, definitions, or topics."""
        try:
            results = llama_client.vector_stores.search(
                vector_store_id=vector_store_id,
                query=query,
                max_num_results=3,
                search_mode="vector"
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

    tools = [
        search_knowledge_base,
        find_professors_by_expertise,
        {
            "type": "mcp",
            "server_label": "canopy-calendar",
            "server_url": config["student-assistant"].get("mcp_calendar_url", "http://canopy-mcp-calendar-mcp-server:8080/sse"),
            "require_approval": "never",
        }
    ]

    llm = ChatOpenAI(
        openai_api_base=config["LLAMA_STACK_URL"] + "/v1",
        model=config["student-assistant"]["model"],
        openai_api_key="not-needed",
        use_responses_api=True,
        temperature=config["student-assistant"].get("temperature", 0.1)
    )

    agent = create_react_agent(
        llm,
        tools,
        prompt=config["student-assistant"].get("prompt", "You are a helpful university assistant."),
        checkpointer=MemorySaver()
    )

class PromptRequest(BaseModel):
    prompt: str

@app.get("/feature-flags")
async def get_feature_flags() -> Dict[str, Any]:
    """Get all feature flags configuration"""
    return FEATURE_FLAGS

@app.post("/summarize")
async def summarize(request: PromptRequest):
    # Check if summarization feature is enabled
    if not FEATURE_FLAGS.get("summarize", False):
        raise HTTPException(status_code=404, detail="Summarization feature is not enabled")

    sys_prompt = config["summarize"].get("prompt", "Summarize the following text:")
    temperature = config["summarize"].get("temperature", 0.7)
    max_tokens = config["summarize"].get("max_tokens", 4096)

    q = queue.Queue()

    def worker():
        print(f"sending requestion to model {config['summarize']['model']}")
        try:
            response = llama_client.chat.completions.create(
                model=config["summarize"]["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": request.prompt},
                ],
                max_tokens=max_tokens, 
                temperature=temperature,
                stream=True,
            )
            for r in response:
                if hasattr(r, 'choices') and r.choices:
                    delta = r.choices[0].delta
                    chunk = f"data: {json.dumps({'delta': delta.content})}\n\n"
                    q.put(chunk)
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

@app.post("/information-search")
async def information_search(request: PromptRequest):
    # Check if information search feature is enabled
    if not FEATURE_FLAGS.get("information-search", False):
        raise HTTPException(status_code=404, detail="Information search feature is not enabled")

    # Dummy information search implementation
    sys_prompt = config["information-search"]["prompt"]
    temperature = config["information-search"].get("temperature", 0.7)
    max_tokens = config["information-search"].get("max_tokens", 4096)
    vector_db_id = config["information-search"].get("vector_db_id", "latest")

    q = queue.Queue()

    print(f"Searching in collection {vector_db_id}")
    print(f"Existing collections: {llama_client.vector_stores.list()}")
    print(f"query: {request.prompt}")

    search_results = llama_client.vector_stores.search(
        vector_store_id=vector_db_id,
        query=request.prompt,
        max_num_results=5,
        search_mode="vector"
    )
    retrieved_chunks = []
    for i, result in enumerate(search_results.data):
        chunk_content = result.content[0].text if hasattr(result, 'content') else str(result)
        retrieved_chunks.append(chunk_content)

    prompt_context = "\n\n".join(retrieved_chunks)

    enhaned_prompt = f"""Please answer the given query using the document intelligence context below.

    CONTEXT (Processed with Docling Document Intelligence):
    {prompt_context}

    QUERY:
    {request.prompt}

    Note: The context includes intelligently processed content with preserved tables, formulas, figures, and document structure."""

    def worker():
        print(f"sending requestion to model {config['summarize']['model']}")
        try:
            response = llama_client.chat.completions.create(
                model=config["summarize"]["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": enhaned_prompt},
                ],
                max_tokens=max_tokens, 
                temperature=temperature,
                stream=True,
            )
            for r in response:
                if hasattr(r, 'choices') and r.choices:
                    delta = r.choices[0].delta
                    chunk = f"data: {json.dumps({'delta': delta.content})}\n\n"
                    q.put(chunk)
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

@app.post("/student-assistant")
async def student_assistant_chat(request: PromptRequest):
    if not FEATURE_FLAGS.get("student-assistant", False):
        raise HTTPException(status_code=404, detail="Student assistant feature is not enabled")

    if not agent:
        raise HTTPException(status_code=500, detail="Student assistant not initialized")

    q = queue.Queue()

    def worker():
        try:
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            thread_id = str(int(random.random() * 1000000))
            config_params = {"configurable": {"thread_id": thread_id}}
            inputs = {"messages": [HumanMessage(content=request.prompt)]}

            seen_messages = 0
            seen_tool_calls = set()  # Track which tool calls we've already sent

            for state in agent.stream(inputs, config_params, stream_mode="values"):
                messages = state.get("messages", [])
                new_messages = messages[seen_messages:]
                seen_messages = len(messages)

                for msg in new_messages:
                    msg_type = getattr(msg, "type", None)

                    if msg_type == "ai":
                        # Check for tool calls
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                # Create unique identifier for this tool call
                                tool_call_id = f"{tc.get('name')}:{json.dumps(tc.get('args'), sort_keys=True)}"
                                if tool_call_id not in seen_tool_calls:
                                    seen_tool_calls.add(tool_call_id)
                                    tool_call_data = {
                                        "type": "tool_call",
                                        "name": tc.get("name"),
                                        "args": tc.get("args")
                                    }
                                    chunk = f"data: {json.dumps(tool_call_data)}\n\n"
                                    q.put(chunk)
                        else:
                            # Check for MCP tool outputs
                            tool_outputs = msg.additional_kwargs.get("tool_outputs", [])
                            for t in tool_outputs:
                                if t.get("type") == "mcp_call":
                                    mcp_data = {
                                        "type": "mcp_call",
                                        "name": t.get("name"),
                                        "server_label": t.get("server_label"),
                                        "arguments": t.get("arguments"),
                                        "output": t.get("output", ""),
                                        "error": t.get("error", "")
                                    }
                                    chunk = f"data: {json.dumps(mcp_data)}\n\n"
                                    q.put(chunk)

                    elif msg_type == "tool":
                        # Tool result
                        tool_result_data = {
                            "type": "tool_result",
                            "name": msg.name,
                            "content": str(msg.content)  # Show more content
                        }
                        chunk = f"data: {json.dumps(tool_result_data)}\n\n"
                        q.put(chunk)

            # After streaming all intermediate steps, send the final answer
            if messages:
                for msg in reversed(messages):
                    if getattr(msg, "type", None) == "ai":
                        content = msg.content
                        if isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))
                            content = "".join(text_parts)

                        if content:
                            # Send final answer marker
                            chunk = f"data: {json.dumps({'type': 'final_answer'})}\n\n"
                            q.put(chunk)
                            # Stream the final answer character by character
                            for char in content:
                                chunk = f"data: {json.dumps({'delta': char})}\n\n"
                                q.put(chunk)
                        break
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

@app.post("/chat-with-shields")
async def chat_with_shields(request: PromptRequest):
    """
    Chat endpoint that uses Llama Stack shields for content moderation.
    Uses the alpha agents API to support input_shields and output_shields.
    """
    if not FEATURE_FLAGS.get("shields", False):
        raise HTTPException(status_code=404, detail="Shields feature is not enabled")

    q = queue.Queue()

    def worker():
        try:
            # Get shields configuration
            input_shields = SHIELDS_CONFIG.get("input_shields", [])
            output_shields = SHIELDS_CONFIG.get("output_shields", [])
            model = SHIELDS_CONFIG.get("model", "llama32")

            # Create agent config with shields using alpha API (llama-stack-client 0.3.0)
            agent_config = {
                "model": model,
                "instructions": config["shields"].get("prompt", "You are a helpful assistant."),
                "sampling_params": {
                    "max_tokens": config["shields"].get("max_tokens", 512),
                    "temperature": config["shields"].get("temperature", 0.7),
                },
                "input_shields": input_shields,
                "output_shields": output_shields,
                "max_infer_iters": 10,
            }

            # Create agent using alpha API
            agent_response = llama_client.alpha.agents.create(agent_config=agent_config)
            agent_id = agent_response.agent_id

            # Create session
            session_response = llama_client.alpha.agents.session.create(
                agent_id=agent_id,
                session_name=f"shields_session_{random.randint(1, 1000000)}"
            )
            session_id = session_response.session_id

            # Send turn with streaming
            response = llama_client.alpha.agents.turn.create(
                agent_id=agent_id,
                session_id=session_id,
                messages=[{"role": "user", "content": request.prompt}],
                stream=True,
            )

            # Process streaming response
            for r in response:
                if hasattr(r, 'event') and hasattr(r.event, 'payload'):
                    payload = r.event.payload

                    # Check for step_progress event with delta (normal streaming)
                    if hasattr(payload, 'event_type') and payload.event_type == 'step_progress':
                        if hasattr(payload, 'delta') and hasattr(payload.delta, 'text'):
                            text_content = payload.delta.text
                            if text_content:
                                chunk = f"data: {json.dumps({'delta': text_content})}\n\n"
                                q.put(chunk)

                    # Check for shield violations (step_complete with violation)
                    elif hasattr(payload, 'event_type') and payload.event_type == 'step_complete':
                        if hasattr(payload, 'step_details'):
                            step_details = payload.step_details
                            if hasattr(step_details, 'step_type') and step_details.step_type == 'shield_call':
                                if hasattr(step_details, 'violation') and step_details.violation is not None:
                                    # Shield violation detected
                                    violation_msg = getattr(step_details.violation, 'user_message', 'Content blocked by safety shields')
                                    chunk = f"data: {json.dumps({'type': 'shield_violation', 'message': violation_msg})}\n\n"
                                    q.put(chunk)
                                    break

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