from logging import LoggerAdapter
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.pipeline import Pipeline
from kink import di


class OverSampler(BaseOverSampler):
    """Oversample minority classes.

    Combines a Random and SMOTE upsampler. Random is required such that
    the neighbors for SMOTE are available.

    Attributes
    ----------
    k_neighbours: int
        The number of neighbors to use for SMOTE.
        The RandomUpsampler with upsample all classes to k_neighbours+1 if needed.
    percentile: float
        The target percentile value to upsample to.
    logger: LoggerAdapter
        The logger to log information.
    ensemble_sampler: Pipeline
        The ensemble sampler to upsample with. Contains RandomOverSampler and SMOTE.
        Random will upsample classes with fewer samples than k_neighbours+1
            to k_neighbours+1.
        SMOTE will upsample classes with fewer samples than the percentile value to the
            percentile value.

    """

    k_neighbours: int
    percentile: float
    logger: LoggerAdapter

    ensemble_sampler: Pipeline

    def __init__(self, k_neighbours: int = 2, percentile: float = 0.5):
        self.k_neighbours = k_neighbours
        self.percentile = percentile
        self.logger = di[LoggerAdapter]
        super().__init__()

    def random_sampler_strategy(self, y: np.ndarray) -> dict:
        """Find the samples to upsample with the RandomOverSampler.

        SMOTE requires that all classes have at least k_neighbors samples.
        We therefor randomly oversample all classes with fewer samples than
        k_neighbors. Result: dict with each class that has < k samples,
        values: k.
        """
        target = self.k_neighbours + 1

        category_counts = pd.Series(y).value_counts()
        categories_to_oversample = category_counts[category_counts < target].index
        return {c: target for c in categories_to_oversample}

    def smote_sampler_strategy(self, y: np.ndarray) -> dict:
        """Find the samples to upsample with SMOTE.

        Find the percentile value. Find all categories with fewer samples than the
        percentile value. Return a dict with each category that has < percentile_value
        samples, values: percentile_value.
        """
        category_counts = pd.Series(y).value_counts()
        percentile_value = ceil(category_counts.quantile(self.percentile))
        categories_to_oversample = category_counts[
            category_counts < percentile_value
        ].index
        return {c: percentile_value for c in categories_to_oversample}

    def upsampled_categories(self, step_name: str) -> list:
        return list(
            self.ensemble_sampler.named_steps[step_name].sampling_strategy_.keys()
        )

    def upsampled_to(self, step_name: str) -> int:
        return next(
            iter(
                self.ensemble_sampler.named_steps[step_name].sampling_strategy_.values()
            ),
            0,
        )

    def _fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **params: Any,  # noqa: ANN401
    ) -> "OverSampler":
        """Fit the OverSampler. Log the upsampled categories."""
        if self.k_neighbours == 0 or self.percentile == 0.0:
            self.logger.debug("No upsampling needed")
            return X, y
        self.ensemble_sampler = Pipeline(
            steps=[
                (
                    "random",
                    RandomOverSampler(sampling_strategy=self.random_sampler_strategy),
                ),
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=self.smote_sampler_strategy,
                        k_neighbors=self.k_neighbours,
                    ),
                ),
            ]
        )
        result = self.ensemble_sampler.fit_resample(X, y, **params)
        for step_name in self.ensemble_sampler.named_steps:
            self.logger.debug(
                "Oversampled categories %s to %s",
                self.upsampled_categories(step_name),
                self.upsampled_to(step_name),
            )
        return result
