"""
FastAPI gateway proxy for vLLM model serving.
Routes requests to the correct vLLM backend based on the 'model' field.
"""
import os
import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def load_routes(config_path: str = "/app/models.yaml") -> dict[str, str]:
    """Load routing table from models.yaml. Returns {model_name: upstream_url}."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    routes = {}
    for name, model in config.get("models", {}).items():
        if model.get("enabled", True):
            routes[name] = f"http://{name}:8000"
    return routes


def create_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="LLM Gateway", version="1.0.0")

    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "/app/models.yaml")

    routes = load_routes(config_path)

    @app.get("/health")
    async def health():
        statuses = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, url in routes.items():
                try:
                    resp = await client.get(f"{url}/health")
                    statuses[name] = "healthy" if resp.status_code == 200 else "unhealthy"
                except httpx.RequestError:
                    statuses[name] = "down"

        healthy_count = sum(1 for s in statuses.values() if s == "healthy")
        total = len(statuses)

        if healthy_count == total:
            status, code = "healthy", 200
        elif healthy_count > 0:
            status, code = "degraded", 200
        else:
            status, code = "unhealthy", 503

        return JSONResponse(status_code=code, content={"status": status, "models": statuses})

    @app.get("/v1/models")
    async def list_models():
        data = [{"id": name, "object": "model", "owned_by": "local"} for name in routes]
        return {"object": "list", "data": data}

    @app.api_route("/v1/{path:path}", methods=["POST"])
    async def proxy(request: Request, path: str):
        body = await request.json()
        model = body.get("model")

        if not model:
            return JSONResponse(status_code=422, content={"error": "Missing 'model' field in request body", "available_models": list(routes.keys())})

        upstream = routes.get(model)
        if not upstream:
            return JSONResponse(status_code=422, content={"error": f"Unknown model: {model}", "available_models": list(routes.keys())})

        upstream_url = f"{upstream}/v1/{path}"
        is_stream = body.get("stream", False)

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            if is_stream:
                req = client.build_request("POST", upstream_url, json=body, headers={"Content-Type": "application/json"})
                resp = await client.send(req, stream=True)

                async def stream_response():
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        await resp.aclose()

                return StreamingResponse(stream_response(), status_code=resp.status_code, media_type="text/event-stream")
            else:
                try:
                    resp = await client.post(upstream_url, json=body, headers={"Content-Type": "application/json"})
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
                except httpx.RequestError:
                    return JSONResponse(status_code=503, content={"error": f"Backend '{model}' is unavailable"})

    return app


def _get_app():
    """Lazy app factory for uvicorn. Only called at container startup."""
    return create_app()
