import warnings

from bunq_ynab_connect.bootstrap import (
    bootstrap_di,
    import_mlserver_windows_friendly,
    monkey_patch_mlserver,
    monkey_patch_ynab,
    setup_database,
)

bootstrap_di()
monkey_patch_ynab()
import_mlserver_windows_friendly()
monkey_patch_mlserver()
setup_database()
# Disable excessive logging from the bunq api
warnings.filterwarnings("ignore", module="bunq")
