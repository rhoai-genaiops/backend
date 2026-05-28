# canopy-be

FastAPI backend service that uses Llamastack to interact with language models.

## Local Development

1. Install dependencies:
```bash
cd app
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## Container Build

Build the container image:
```bash
cd app
podman build -t canopy-backend .
```

## OpenShift Deployment

Deploy using Helm:
```bash
helm install canopy-backend ./helm-chart/canopy-backend
```

## API Endpoints

### POST /summarization

Summarize text using the language model with streaming response.

Request body:
```json
{
    "prompt": "Your text to summarize here"
}
```

Response: Server-sent events stream with delta chunks

## API Documentation

Once the server is running, access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`