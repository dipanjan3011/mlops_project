"""
FastAPI application for serving churn predictions.

Endpoints:
- POST /predict         -- Single customer churn prediction
- POST /batch-predict   -- Batch predictions for multiple customers
- GET  /health          -- Health check with model status
- GET  /metrics         -- Prometheus metrics (scraped by Prometheus)

The app loads the champion model from MLflow on startup. If the model
is not yet available (first boot before training), it returns 503 on
prediction endpoints until a model is loaded.
"""
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from serving.schemas import (
    CustomerInput,
    CustomerIDInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse,
)
from serving.middleware import (
    metrics_middleware,
    record_prediction,
    update_model_info,
)
from models.predict import ChurnPredictor
from features.feature_engineering import compute_features
from features.feast_client import get_online_features


# Global predictor instance
predictor = ChurnPredictor(
    tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup with retry logic.

    Runs in background so the API starts accepting health checks immediately,
    even if the model isn't ready yet. Prediction endpoints return 503 until loaded.
    """
    # Try to load model in background -- don't block startup
    async def load_model_background():
        """Attempt model loading with exponential backoff."""
        max_attempts = 5
        for attempt in range(max_attempts):
            success = predictor.load_model(max_retries=1)
            if success:
                update_model_info(
                    model_name=os.getenv("MODEL_NAME", "churn-model"),
                    model_version=str(predictor.model_version),
                )
                print(f"Model loaded successfully: v{predictor.model_version}")
                return
            wait = min(2 ** (attempt + 1), 30)
            print(f"Model not available yet (attempt {attempt + 1}/{max_attempts}). Retrying in {wait}s...")
            await asyncio.sleep(wait)
        print("WARNING: Could not load model. Prediction endpoints will return 503.")

    asyncio.create_task(load_model_background())
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict customer churn using XGBoost model served via MLflow",
    version="1.0.0",
    lifespan=lifespan,
)

# Add Prometheus metrics middleware
app.middleware("http")(metrics_middleware)


@app.post("/predict", response_model=PredictionOutput)
async def predict(customer: CustomerInput):
    """Predict churn for a single customer.

    Takes raw customer attributes, computes features on-the-fly,
    and returns the churn prediction with probability.

    Returns 503 if no model is loaded yet.
    """
    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again later.",
        )

    try:
        # Compute features from raw input
        features = compute_features(customer.model_dump(exclude={"customerID"}))

        # Make prediction
        result = predictor.predict(features)

        # Record metrics for monitoring
        record_prediction(result["churn_prediction"], result["churn_probability"])

        return PredictionOutput(
            customerID=customer.customerID,
            churn_prediction=result["churn_prediction"],
            churn_probability=result["churn_probability"],
            model_version=result["model_version"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/feast", response_model=PredictionOutput)
async def predict_from_feast(request: CustomerIDInput):
    """Predict churn by looking up features from the Feast online store (Redis).

    Only requires a customerID — all features are fetched from Redis.
    Features must have been materialized first via `make feast-materialize`.

    Returns 503 if no model is loaded, 404 if customer not found in Feast.
    """
    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again later.",
        )

    try:
        # Fetch features from Feast online store (Redis)
        feast_result = get_online_features([request.customerID])

        # Check if Feast returned actual values (None means not materialized)
        if feast_result.get("tenure", [None])[0] is None:
            raise HTTPException(
                status_code=404,
                detail=f"Customer '{request.customerID}' not found in feature store. "
                       "Run 'make feast-materialize' to load features into Redis.",
            )

        # Convert Feast dict to a single-row dict for compute_features
        raw_features = {k: v[0] for k, v in feast_result.items() if k != "customerID"}

        # Compute engineered features (tenure_bucket, service_count, etc.)
        features = compute_features(raw_features)

        # Make prediction
        result = predictor.predict(features)

        # Record metrics
        record_prediction(result["churn_prediction"], result["churn_probability"])

        return PredictionOutput(
            customerID=request.customerID,
            churn_prediction=result["churn_prediction"],
            churn_probability=result["churn_probability"],
            model_version=result["model_version"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feast prediction failed: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionOutput)
async def batch_predict(batch: BatchPredictionInput):
    """Predict churn for multiple customers in a single request.

    Returns 503 if no model is loaded yet.
    """
    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again later.",
        )

    try:
        predictions = []
        for customer in batch.customers:
            features = compute_features(customer.model_dump(exclude={"customerID"}))
            result = predictor.predict(features)
            record_prediction(result["churn_prediction"], result["churn_probability"])
            predictions.append(
                PredictionOutput(
                    customerID=customer.customerID,
                    churn_prediction=result["churn_prediction"],
                    churn_probability=result["churn_probability"],
                    model_version=result["model_version"],
                )
            )

        return BatchPredictionOutput(predictions=predictions, count=len(predictions))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint.

    Returns model loading status. Used by Docker health checks and
    load balancers to determine if the service is ready.
    """
    return HealthResponse(
        status="healthy" if predictor.model is not None else "degraded",
        model_loaded=predictor.model is not None,
        model_version=predictor.model_version,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint.

    Returns all registered Prometheus metrics in the exposition format.
    Scraped by Prometheus every 15s (configured in prometheus.yml).
    """
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
