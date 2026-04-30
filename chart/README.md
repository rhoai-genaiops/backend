# Canopy Backend Helm Chart

A Helm chart for deploying the Canopy FastAPI Backend application on Kubernetes/OpenShift.

## Overview

This chart deploys the Canopy Backend application, which is a FastAPI-based service that provides LLM integration capabilities with configurable feature flags for summarization, information search, student assistant, Socratic tutoring, and more.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- OpenShift 4.x+ (for Route resource)

## Installation

### Install the chart

```bash
helm install canopy-backend ./chart
```

### Install with custom values

```bash
helm install canopy-backend ./chart -f custom-values.yaml
```

### Upgrade the chart

```bash
helm upgrade canopy-backend ./chart
```

## Configuration

Each feature declares its own `endpoint` — the URL of the model server (e.g. vLLM) or LLaMA Stack instance it connects to. This allows different features to target different backends independently.

### Global Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `https://mlflow.redhat-ods-applications.svc.cluster.local:8443` |

### Summarize

| Parameter | Description | Default |
|-----------|-------------|---------|
| `summarization.enabled` | Enable summarization functionality | `true` |
| `summarization.endpoint` | Model endpoint URL | `http://llama-32-predictor:80/v1` |
| `summarization.model` | Model identifier | `llama32` |
| `summarization.temperature` | Temperature for generation | `0.9` |
| `summarization.max_tokens` | Max tokens for generation | `4096` |
| `summarization.mlflow_prompt` | MLflow prompt registry name | `summarization` |
| `summarization.mlflow_prompt_version` | MLflow prompt version | `latest` |
| `summarization.prompt_b` | Alternative prompt for A/B testing | - |

### Information Search

| Parameter | Description | Default |
|-----------|-------------|---------|
| `information-search.enabled` | Enable information search | `false` |
| `information-search.endpoint` | LLaMA Stack endpoint URL | `http://llama-stack-service:8321/v1` |
| `information-search.model` | Model identifier | `llama32` |
| `information-search.temperature` | Temperature for generation | `0.7` |
| `information-search.max_tokens` | Max tokens for generation | `4096` |
| `information-search.vector_db_id` | Vector database identifier | `latest` |
| `information-search.mlflow_prompt` | MLflow prompt registry name | `information-search` |

### Shields

| Parameter | Description | Default |
|-----------|-------------|---------|
| `shields.enabled` | Enable input/output moderation | `false` |
| `shields.endpoint` | LLaMA Stack endpoint URL | `http://llama-stack-service:8321` |
| `shields.input_shields` | Shield names for input moderation | `[]` |
| `shields.output_shields` | Shield names for output moderation | `[]` |
| `shields.check_interval` | Interval for shield checks | `50` |

### Student Assistant

| Parameter | Description | Default |
|-----------|-------------|---------|
| `student-assistant.enabled` | Enable student assistant | `false` |
| `student-assistant.endpoint` | LLaMA Stack endpoint URL | `http://llama-stack-service:8321/v1` |
| `student-assistant.model` | Model identifier | `llama32` |
| `student-assistant.temperature` | Temperature for generation | `0.1` |
| `student-assistant.vector_db_id` | Vector database identifier | `latest` |
| `student-assistant.mcp_calendar_url` | MCP calendar server URL | `http://canopy-mcp-calendar-mcp-server:8080/sse` |
| `student-assistant.mlflow_prompt` | MLflow prompt registry name | `student-assistant` |

### Socratic Tutor

| Parameter | Description | Default |
|-----------|-------------|---------|
| `socratic-tutor.enabled` | Enable Socratic tutor | `false` |
| `socratic-tutor.endpoint` | Model endpoint URL | `http://llama-32-predictor:80/v1` |
| `socratic-tutor.model` | Model identifier | `llama32` |
| `socratic-tutor.temperature` | Temperature for generation | `0.9` |
| `socratic-tutor.max_tokens` | Max tokens for generation | `1500` |
| `socratic-tutor.mlflow_prompt` | MLflow prompt registry name | `socratic-tutor` |

### Feedback

| Parameter | Description | Default |
|-----------|-------------|---------|
| `feedback.enabled` | Enable feedback collection | `true` |

### A/B Testing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ab_testing.enabled` | Enable A/B testing | `false` |

## Values Structure

The chart uses a structured values file that gets mounted as a ConfigMap. The entire values structure is available to the application as a configuration file at `/canopy/canopy-config.yaml`.

A JSON Schema (`values.schema.json`) is provided for validation.

### Environment Variables

The deployment sets the following environment variables:

- `MLFLOW_TRACKING_URI`: MLflow tracking server URI
- `MLFLOW_TRACKING_INSECURE_TLS`: Set to `true` for in-cluster MLflow

## Resources Created

This chart creates the following Kubernetes resources:

- **Deployment**: Main application deployment with single replica
- **Service**: ClusterIP service exposing port 8000
- **ConfigMap**: Contains the entire values structure as `canopy-config.yaml`
- **ServiceAccount**: Pod identity for RBAC
- **RoleBinding**: Cross-namespace access for prompt registry

## Examples

### Basic Installation (summarization only)

```yaml
# custom-values.yaml
MLFLOW_TRACKING_URI: "https://mlflow.apps.example.com"

summarization:
  enabled: true
  endpoint: "http://llama-32-predictor:80/v1"
  model: llama32
  mlflow_prompt: summarization

feedback:
  enabled: true
```

### Adding Llama Stack features

```yaml
# custom-values.yaml
MLFLOW_TRACKING_URI: "https://mlflow.apps.example.com"

summarization:
  enabled: true
  endpoint: "http://llama-32-predictor:80/v1"
  model: llama32
  mlflow_prompt: summarization

information-search:
  enabled: true
  endpoint: "http://llama-stack-service:8321/v1"
  model: llama32
  vector_db_id: latest
  mlflow_prompt: information-search

student-assistant:
  enabled: true
  endpoint: "http://llama-stack-service:8321/v1"
  model: llama32
  vector_db_id: latest
  mlflow_prompt: student-assistant

feedback:
  enabled: true
```

## Uninstallation

```bash
helm uninstall canopy-backend
```

## Support

For issues and questions, please refer to the project documentation or open an issue in the project repository.

Helm chart icon is from [here](https://www.deviantart.com/pratlegacy/art/Cute-Groot-Digital-Art-Vector-Icon-762435201)
