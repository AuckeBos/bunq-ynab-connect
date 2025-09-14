sudo "./provide_docker_access.sh"
cd "/home/bunqynab/bunq_ynab_connect"
uv run prefect concurrency-limit create single-payment-sync 1
uv run python "worker.py"