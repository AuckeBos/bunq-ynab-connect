from logging import LoggerAdapter
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from kink import di


class UnderSampler(RandomUnderSampler):
    """Undersample the majority classes to the value of a quantile.

    Attributes
    ----------
    percentile: float
        The percentile value to undersample to.
    logger: LoggerAdapter
        The logger to log information.

    """

    percentile: float
    logger: LoggerAdapter

    def __init__(self, percentile: float = 0.9):
        self.percentile = percentile
        self.logger = di[LoggerAdapter]
        super().__init__(sampling_strategy=self.sampling_strategy)

    def sampling_strategy(self, y: np.ndarray) -> dict:
        """Return the undersampling strategy for the UnderSampler.

        Undersample the categories that have a count higher than the quantile value.
        Undersample to the value of the quantile value.
        """
        category_counts = pd.Series(y).value_counts()
        percentile_value = ceil(category_counts.quantile(self.percentile))

        categories_to_undersample = category_counts[
            category_counts > percentile_value
        ].index

        return {c: percentile_value for c in categories_to_undersample}

    @property
    def undersampled_categories(self) -> list:
        return list(self.sampling_strategy_.keys())

    @property
    def undersampled_to(self) -> int:
        return next(iter(self.sampling_strategy_.values()), 0)

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **params: Any,
    ) -> "UnderSampler":
        """Fit the UnderSampler. Log the undersampled categories."""
        result = super().fit_resample(X, y, **params)
        categories = self.undersampled_categories
        self.logger.debug(
            "Undersampled categories %s to %s", categories, self.undersampled_to
        )
        return result
