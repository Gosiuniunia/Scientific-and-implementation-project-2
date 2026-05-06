venv:
    source .venv/bin/activate

test:
    pytest tests/ --cov=core --cov-report=term-missing

run-app:
    python3 run_pcoa_app.py