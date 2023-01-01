from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from demo_code.MStep import m_step

if TYPE_CHECKING:
    from src.models.binary_latent_factor_model_approximation import (
        BinaryLatentFactorApproximation,
    )


class AbstractBinaryLatentFactorModel(ABC):
    @property
    @abstractmethod
    def mu(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def variance(self) -> float:
        pass

    @property
    @abstractmethod
    def pi(self) -> np.ndarray:
        pass

    @abstractmethod
    def maximisation_step(
        self,
        x: np.ndarray,
        binary_latent_factor_approximation: BinaryLatentFactorApproximation,
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


class BinaryLatentFactorModel(AbstractBinaryLatentFactorModel):
    """
    mu: matrix of means (number_of_dimensions, number_of_latent_variables)
    sigma: gaussian noise parameter
    pi: vector of priors (1, number_of_latent_variables)
    """

    def __init__(
        self,
        mu: np.ndarray,
        sigma: float,
        pi: np.ndarray,
    ):
        self._mu = mu
        self._sigma = sigma
        self._pi = pi

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    @property
    def pi(self):
        return self._pi

    @pi.setter
    def pi(self, value):
        self._pi = value

    @property
    def variance(self) -> float:
        return self.sigma**2

    @staticmethod
    def calculate_maximisation_parameters(
        x: np.ndarray,
        binary_latent_factor_approximation: BinaryLatentFactorApproximation,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        return m_step(
            x=x,
            es=binary_latent_factor_approximation.expectation_s,
            ess=binary_latent_factor_approximation.expectation_ss,
        )

    def maximisation_step(
        self,
        x: np.ndarray,
        binary_latent_factor_approximation: BinaryLatentFactorApproximation,
    ) -> None:
        mu, sigma, pi = self.calculate_maximisation_parameters(
            x, binary_latent_factor_approximation
        )
        self.mu = mu
        self.sigma = sigma
        self.pi = pi


def init_binary_latent_factor_model(
    x: np.ndarray,
    binary_latent_factor_approximation: BinaryLatentFactorApproximation,
) -> BinaryLatentFactorModel:
    mu, sigma, pi = BinaryLatentFactorModel.calculate_maximisation_parameters(
        x, binary_latent_factor_approximation
    )
    return BinaryLatentFactorModel(mu, sigma, pi)
