venv:
    source .venv/bin/activate

test:
    pytest tests/ --cov=core --cov-report=term-missing

run-app:
    python run_pcoa_app.py

launch-microservice:
    uvicorn core.microservice:app --reload

launch-app:
    uvicorn core.microservice:app --reload &
    python run_pcoa_app.py