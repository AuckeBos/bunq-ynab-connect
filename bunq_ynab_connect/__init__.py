import warnings

from bunq_ynab_connect.bootstrap import (
    bootstrap_di,
    import_mlserver_windows_friendly,
    monkey_patch_mlserver,
    monkey_patch_ynab,
)

bootstrap_di()
monkey_patch_ynab()
import_mlserver_windows_friendly()
monkey_patch_mlserver()
# Disable excessive logging from the bunq api
warnings.filterwarnings("ignore", module="bunq")

# Disable warning about predicting classes not in the training set
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.metrics",
    message="y_pred contains classes not in y_true",
)

# Disable warnings about classes smaller than n_splits
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.model_selection",
    message="The least populated class in y has only",
)
