# Introduction
[Kink](https://github.com/kodemore/kink) is used as dependency injection framework. [bootstrap.py](/bunq_ynab_connect/bootstrap.py) injects all dependencies that cannot be automatically injected. This script is imported in the main `__init__.py` file of the package. It runs some other setup, like monkey patching the Ynab SDK, and setting up the Prefect logger.
