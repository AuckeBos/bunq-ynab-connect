sudo "./provide_docker_access.sh"
cd "/home/bunqynab/bunq_ynab_connect"
prefect gcl create single-payment-sync --limit 1
python "worker.py"