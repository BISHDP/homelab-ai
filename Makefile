.PHONY: start stop status dashboard

VENV = .venv/bin

start:
	docker compose -f docker/jupyter/docker-compose.yml up -d

stop:
	docker compose -f docker/jupyter/docker-compose.yml down

status:
	systemctl --user status homelab-collector
	docker ps

collector-start:
	systemctl --user start homelab-collector

collector-stop:
	systemctl --user stop homelab-collector

dashboard:
	$(VENV)/streamlit run monitor/dashboard/app.py