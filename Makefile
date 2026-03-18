.PHONY: setup docker-up docker-down train serve test lint feast-apply feast-materialize clean

# ——— Setup ———
setup:
	cp -n .env.example .env || true
	pip install -r requirements.txt
	pip install -e .

# ——— Docker ———
docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps

# ——— Data & Training ———
train:
	python -m models.train

preprocess:
	python -m data.load

validate:
	python -m validation.validate

# ——— Feature Store ———
feast-apply:
	docker compose exec api python -c "\
		import importlib.util, os; \
		os.chdir('/app/config/feast'); \
		spec = importlib.util.spec_from_file_location('features', '/app/config/feast/features.py'); \
		mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); \
		from feast import FeatureStore; store = FeatureStore(repo_path='/app/config/feast'); \
		store.apply([mod.customer, mod.customer_demographics, mod.customer_account, mod.customer_services]); \
		print('Feast feature views applied')"

feast-materialize:
	docker compose exec api python -c "\
		from features.feast_client import get_store; \
		from datetime import datetime, timedelta; \
		store = get_store(); \
		store.materialize(start_date=datetime.now() - timedelta(days=365), end_date=datetime.now() + timedelta(days=2)); \
		print('Materialization complete')"

# ——— Serving ———
serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# ——— Testing ———
test:
	PYTHONPATH=src pytest tests/ -v

test-unit:
	PYTHONPATH=src pytest tests/unit/ -v

test-integration:
	PYTHONPATH=src pytest tests/integration/ -v

# ——— Linting ———
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/

# ——— Cleanup ———
clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
