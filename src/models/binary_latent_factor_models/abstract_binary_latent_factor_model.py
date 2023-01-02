from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.binary_latent_factor_approximations.abstract_binary_latent_factor_approximation import (
        AbstractBinaryLatentFactorApproximation,
    )


class AbstractBinaryLatentFactorModel(ABC):
    @property
    @abstractmethod
    def mu(self) -> np.ndarray:
        """
        matrix of means (number_of_dimensions, number_of_latent_variables)
        """
        pass

    @property
    @abstractmethod
    def variance(self) -> float:
        """
        gaussian noise parameter
        """
        pass

    @property
    @abstractmethod
    def pi(self) -> np.ndarray:
        """
        (1, number_of_latent_variables)
        """
        pass

    @abstractmethod
    def maximisation_step(
        self,
        x: np.ndarray,
        binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
    ) -> None:
        pass

    def mu_exclude(self, exclude_latent_index: int) -> np.ndarray:
        return np.concatenate(  # (number_of_dimensions, number_of_latent_variables-1)
            (self.mu[:, :exclude_latent_index], self.mu[:, exclude_latent_index + 1 :]),
            axis=1,
        )

    @property
    def log_pi(self) -> np.ndarray:
        return np.log(self.pi)

    @property
    def log_one_minus_pi(self) -> np.ndarray:
        return np.log(1 - self.pi)

    @property
    def precision(self) -> float:
        return 1 / self.variance

    @property
    def d(self) -> int:
        return self.mu.shape[0]

    @property
    def k(self) -> int:
        return self.mu.shape[1]
