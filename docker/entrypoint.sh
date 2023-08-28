cd bunq_ynab_connect
prefect deploy --name extract
prefect worker start -p docker-pool