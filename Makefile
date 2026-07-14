.PHONY: build up shell smoke test test-sim eval-mini pull-model

build:
	docker compose build nav

up:
	docker compose up -d ollama

pull-model:
	docker compose exec ollama ollama pull qwen2.5vl:3b

shell:
	docker compose run --rm nav bash

smoke:
	docker compose run --rm nav python scripts/smoke_habitat.py
	docker compose run --rm nav python scripts/smoke_ollama.py

test:
	docker compose run --rm nav pytest tests/unit -q

test-sim:
	docker compose run --rm nav pytest tests/integration -q -m sim

eval-mini:
	docker compose run --rm nav python scripts/run_eval.py eval=hm3d_val_mini
