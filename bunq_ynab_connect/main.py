from bunq_ynab_connect.data.data_extractors.bunq_extractor import BunqExtractor
from bunq_ynab_connect.data.data_extractors.ynab_account_extractor import (
    YnabAccountExtractor,
)
from bunq_ynab_connect.data.data_extractors.ynab_budget_extractor import (
    YnabBudgetExtractor,
)

extractor = YnabAccountExtractor()
extractor.extract()
