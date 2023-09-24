# Provide docker access to the user
sudo ./provide_docker_access.sh
# Start the prefect agent
cd bunq_ynab_connect
prefect deploy --all
prefect worker start -p docker-pool