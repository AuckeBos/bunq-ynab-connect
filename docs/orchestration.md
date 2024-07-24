# Introduction
Prefect is used for orchestration. Any code that should run periodically, should be deployed as a Prefect flow. Moreover, any ad-hoc flows which should be easily manually triggered, should also be deployed as a Prefect flow. The prefect frontend is used as entrypoint to the production server. 

# Deployments
All flows are defined in [flows.py](/bunq_ynab_connect/flows.py). All deployments use this file as entrypoint. The deployments themselves are defined in [prefect.yaml](/bunq_ynab_connect/prefect.yaml). The `prefect-agent` deploys it to the server upon boot. The following deployments exist:

- `extract`. Does not run automatically. Extracts all unextracted payments from bunq. The payments are added to the payment queue, which can be processed using the PaymentSyncer, with flow `sync_payement_queue`.
- `sync_payment_queue`. Does not run automatically. Syncs all unsynced payments in the payment queue to YNAB.
- `sync`. Runs hourly between hours 6 and 23. Runs `extract` and `sync_payment_queue` in sequence.
- `train`. Runs on sunday at 02:00. Trains one model for each budget. Before doing so, extracts all payments, and maps them to transactions in Ynab. Runs 2 experiments. The first one selects the model that fits best with some default params. The second one selects the best configuration with a grid search. Deploys the model to MLServer if it performs better than the current model. Note that the trained model is available the next day, since the [MLServer restarter](/docs/infrastructure.md#mlserver-restarter) restarts the MLServer container daily.
- `sync_payment`. A flow to debug the syncing of a single payment. Can be triggered manually.
- `exchange_pat`. A flow to exchange the Bunq PAT token for a config file. Should only be used if the external IP of the server has changed, and the Bunq PAT is restricted to the old IP. If this is the case:
    - In the Bunq app, update the existing token to allow all IP's. Needed to be able to use it once more.
    - Create a new PAT.
    - Run this flow, with the PAT as parameter.
    - The new PAT is exchanged for a config file, which is restricted to the new IP.
    - Remove the old PAT from the Bunq app.

# Logging
Prefect is used to store the logs. If anything fails or doesn't work as expected, use Prefect as the first source of information. 
