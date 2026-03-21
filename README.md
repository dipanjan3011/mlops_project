# Telco Customer Churn Prediction — MLOps Demo

End-to-end MLOps pipeline for predicting customer churn using the Telco Customer Churn dataset. Built with 100% open-source tools, deployable with a single `docker-compose up`.

> **Want to recreate this project from scratch?** See the step-by-step [Project Recreation Guide](docs/PROJECT_RECREATION_GUIDE.md) — it explains every file, every concept, and every design decision.

## Architecture

```
                    GitHub Actions (CI/CD/CT)
                           │
    ┌──────────────────────┼──────────────────────────┐
    │                      ▼                          │
    │  ┌─────────┐   ┌──────────┐   ┌─────────────┐  │
    │  │  DVC    │──▶│ Airflow  │──▶│   MLflow     │  │
    │  │ (data)  │   │ (DAGs)   │   │ (tracking +  │  │
    │  └─────────┘   └────┬─────┘   │  registry)   │  │
    │                     │         └──────┬──────┘  │
    │  ┌─────────┐        │                │         │
    │  │  Great  │◀───────┘         ┌──────▼──────┐  │
    │  │  Expect │                  │  FastAPI    │  │
    │  └─────────┘   ┌─────────┐   │  (serving)  │  │
    │                │  Feast   │──▶│  :8000      │  │
    │                │ (features│   └──────┬──────┘  │
    │                │  Redis)  │          │         │
    │                └─────────┘   ┌──────▼──────┐  │
    │                              │ Prometheus  │  │
    │  ┌─────────┐                 │  + Grafana  │  │
    │  │Evidently│────────────────▶│ (monitoring)│  │
    │  │ (drift) │                 └─────────────┘  │
    │  └─────────┘                                   │
    └────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Tool |
|---|---|
| Feature Store | Feast (file offline + Redis online) |
| Experiment Tracking | MLflow (PostgreSQL backend) |
| Orchestration | Apache Airflow (LocalExecutor) |
| Model Serving | FastAPI + Uvicorn |
| ML Monitoring | Evidently AI |
| Observability | Prometheus + Grafana |
| CI/CD | GitHub Actions |
| Data Versioning | DVC |
| Data Validation | Great Expectations |
| ML Libraries | scikit-learn, XGBoost |
| Containerization | Docker + Docker Compose |

## Quick Start

### Prerequisites

- Docker Desktop (4GB+ RAM allocated)
- Python 3.11+
- Git

### 1. Clone and Setup

```bash
git clone <repo-url>
cd mlops_project
make setup
```

### 2. Start All Services

```bash
make docker-up
```

This starts 9 services:

| Service | URL | Credentials |
|---|---|---|
| MLflow | http://localhost:5001 | — |
| Airflow | http://localhost:8080 | admin / admin |
| FastAPI | http://localhost:8000 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

> **Note**: MLflow uses port 5001 externally because macOS AirPlay Receiver occupies port 5000. You can free port 5000 by disabling AirPlay Receiver in System Settings → General → AirDrop & Handoff.

### 3. Train the Model

```bash
make train
```

Or trigger the training pipeline in Airflow UI.

### 4. Make Predictions

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```

### 5. Feature Store

```bash
make feast-apply        # Register feature definitions
make feast-materialize  # Push features to Redis
```

## Project Structure

```
mlops_project/
├── .github/workflows/     # CI/CD pipelines
│   ├── test.yml           # Lint + unit/integration tests
│   ├── build.yml          # Docker build + smoke test
│   └── ct.yml             # Continuous training (weekly)
├── config/
│   ├── feast/             # Feature store config
│   ├── grafana/           # Dashboard provisioning
│   ├── prometheus/        # Scrape config
│   └── evidently/         # Drift detection config
├── dags/                  # Airflow DAGs
│   ├── training_pipeline.py
│   ├── continuous_training.py
│   └── feature_materialization.py
├── data/
│   ├── raw/               # Original CSV dataset
│   └── processed/         # Parquet train/test splits
├── docker/                # Dockerfiles
├── notebooks/             # Demo walkthrough
├── src/
│   ├── data/              # Loading + preprocessing
│   ├── features/          # Feast client + feature engineering
│   ├── models/            # Training, evaluation, prediction
│   ├── monitoring/        # Drift detection + metrics
│   ├── serving/           # FastAPI app
│   └── validation/        # Great Expectations suites
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── docker-compose.yml     # Full stack (9 services)
├── Makefile               # Dev shortcuts
├── dvc.yaml               # Data pipeline stages
└── requirements.txt       # Pinned dependencies
```

## Makefile Commands

| Command | Description |
|---|---|
| `make setup` | Install dependencies and create .env |
| `make docker-up` | Start all Docker services |
| `make docker-down` | Stop all services |
| `make train` | Run model training |
| `make test` | Run all tests |
| `make test-unit` | Run unit tests only |
| `make lint` | Run linter |
| `make feast-apply` | Register Feast features |
| `make feast-materialize` | Materialize features to Redis |
| `make clean` | Remove containers, volumes, caches |

## Monitoring

### Grafana Dashboard

The auto-provisioned dashboard at http://localhost:3000 shows:
- API request rate and latency (P50/P95/P99)
- Prediction distribution (churn vs no-churn)
- Model performance metrics (F1, accuracy, AUC)
- Data drift scores per feature

### Drift Detection

Evidently AI monitors for data drift by comparing incoming predictions against the training data distribution. When drift exceeds the threshold (30% of features), the continuous training DAG automatically triggers retraining.

## Troubleshooting

### Docker Memory
The stack requires 4GB+ RAM. Check Docker Desktop → Settings → Resources.

### Port Conflicts
| Port | Service | Common Conflict |
|---|---|---|
| 5001 | MLflow | macOS AirPlay (5000) |
| 8080 | Airflow | Other web servers |
| 3000 | Grafana | Other dev tools |

### Services Not Starting
```bash
# Check service status
docker compose ps

# View logs for a specific service
docker compose logs mlflow
docker compose logs airflow-scheduler

# Restart everything
make docker-down && make docker-up
```

## Testing

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests (requires Docker services running)
make test-integration
```

## CI/CD Pipelines

1. **test.yml** — Runs on PR/push: lint → unit tests → integration tests
2. **build.yml** — Runs on main push: build Docker images → smoke test
3. **ct.yml** — Weekly scheduled: drift check → retrain if needed

## License

MIT
