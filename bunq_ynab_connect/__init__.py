import warnings

from bunq_ynab_connect.bootstrap import bootstrap_di, monkey_patch_ynab

bootstrap_di()
monkey_patch_ynab()
# Disable excessive logging from the bunq api
warnings.filterwarnings("ignore", module="bunq")
