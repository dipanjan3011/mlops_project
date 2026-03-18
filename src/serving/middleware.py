"""
Prometheus metrics middleware for the FastAPI serving layer.

Tracks:
- Total request count by endpoint and status code
- Request latency histogram by endpoint
- Prediction distribution (churn vs no-churn counts)
- Current model version info gauge
"""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Info


# === Prometheus Metrics ===

# Request metrics
REQUEST_COUNT = Counter(
    "api_request_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total number of predictions made",
    ["prediction"],  # "churn" or "no_churn"
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of churn probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Model info
MODEL_INFO = Info(
    "model",
    "Currently loaded model information",
)


def record_prediction(prediction: int, probability: float):
    """Record a prediction in Prometheus metrics."""
    label = "churn" if prediction == 1 else "no_churn"
    PREDICTION_COUNT.labels(prediction=label).inc()
    PREDICTION_PROBABILITY.observe(probability)


def update_model_info(model_name: str, model_version: str):
    """Update the model info gauge."""
    MODEL_INFO.info(
        {
            "name": model_name,
            "version": str(model_version),
        }
    )


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """FastAPI middleware that tracks request count and latency.

    Wraps every request to record:
    - Response status code
    - Request duration
    Excludes /metrics endpoint to avoid self-referential tracking.
    """
    # Skip tracking for the metrics endpoint itself
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    return response
