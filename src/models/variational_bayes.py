from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.models.binary_latent_factor_model import (
    AbstractBinaryLatentFactorModel,
    BinaryLatentFactorApproximation,
)


@dataclass
class Distribution(ABC):
    @property
    @abstractmethod
    def mean(self):
        pass


@dataclass
class Beta(Distribution):
    alpha: np.ndarray
    beta: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return np.divide(self.alpha, (self.alpha + self.beta))


@dataclass
class InverseGamma(Distribution):
    a: float
    b: float

    @property
    def mean(self) -> float:
        return self.b / (self.a - 1)


@dataclass
class Gaussian(Distribution):
    mu: np.ndarray  # (number_of_dimensions,  number_of_latent_variables)
    variance: np.ndarray  # (number_of_latent_variables, )

    @property
    def precision(self) -> np.ndarray:
        return 1 / self.variance

    @property
    def mean(self) -> np.ndarray:
        return self.mu


class VariationalBayesBinaryLatentFactorModel(AbstractBinaryLatentFactorModel):
    def __init__(self, mu: Gaussian, variance: InverseGamma, pi: Beta):
        self._mu = mu
        self._variance = variance
        self._pi = pi

    @property
    def variance(self) -> float:
        return self._variance.mean

    @property
    def pi(self) -> np.ndarray:
        return self._pi.mean

    @property
    def mu(self) -> np.ndarray:
        return self._mu.mean

    def _update_pi(
        self,
        binary_latent_factor_approximation: BinaryLatentFactorApproximation,
    ):
        self._pi.alpha += np.sum(
            binary_latent_factor_approximation.expectation_s, axis=0
        ).reshape(1, -1)
        self._pi.beta += binary_latent_factor_approximation.n - np.sum(
            binary_latent_factor_approximation.expectation_s, axis=0
        ).reshape(1, -1)

    def _update_variance(
        self,
        x: np.ndarray,  # (number_of_points, number_of_dimensions)
        binary_latent_factor_approximation: BinaryLatentFactorApproximation,
    ):
        #  expectation_s (number_of_points, number_of_latent_variables)
        self._variance.a += (
            0.5
            * binary_latent_factor_approximation.n
            * binary_latent_factor_approximation.k
        )
        # self._variance.b += 0.5 * np.mean(
        #     (x - binary_latent_factor_approximation.expectation_s @ self.mu.T) ** 2
        # )
        self._variance.b += 2 * np.sum(
            x - binary_latent_factor_approximation.expectation_s @ self.mu.T
        )

    def _update_mu_k(
        self,
        x: np.ndarray,  # (number_of_points, number_of_dimensions)
        binary_latent_factor_approximation: BinaryLatentFactorApproximation,
        k: int,  # latent dimension
    ):
        #  expectation_s (number_of_points, number_of_latent_variables)
        #  expectation_ss (number_of_latent_variables, number_of_latent_variables)
        self._mu.variance[k] = 1 / (
            self._mu.precision[k]
            + np.mean(binary_latent_factor_approximation.expectation_s[:, k])
            * self.precision
        )

        # (number_of_points, number_of_latent_variables-1)
        es_except_k = np.concatenate(
            (
                binary_latent_factor_approximation.expectation_s[:, :k],
                binary_latent_factor_approximation.expectation_s[:, k + 1 :],
            ),
            axis=1,
        )

        # (number_of_dimensions, number_of_latent_variables-1)
        mu_except_k = np.concatenate((self.mu[:, :k], self.mu[:, k + 1 :]), axis=1)

        # (number_of_dimensions x 1)
        self._mu.mu[:, k] = self._mu.variance[k] * (
            (
                x  # (number_of_points, number_of_dimensions)
                - es_except_k  # (number_of_points, number_of_latent_variables-1)
                @ mu_except_k.T  # (number_of_dimensions, number_of_latent_variables-1)
            ).T  # (number_of_dimensions, number_of_points)
            @ binary_latent_factor_approximation.expectation_s[
                :, k
            ]  # (number_of_points, 1)
        )

    def maximisation_step(
        self,
        x: np.ndarray,
        binary_latent_factor_approximation: BinaryLatentFactorApproximation,
    ) -> None:
        self._update_pi(binary_latent_factor_approximation)
        self._update_variance(x, binary_latent_factor_approximation)
        # es = binary_latent_factor_approximation.expectation_s
        # ess = binary_latent_factor_approximation.expectation_ss
        # n = binary_latent_factor_approximation.n
        # self._pi = np.mean(es, axis=0, keepdims=True)
        # self._sigma = np.sqrt((np.trace(np.dot(x.T, x)) + np.trace(np.dot(np.dot(self.mu.T, self.mu), ess))
        #                  - 2 * np.trace(np.dot(np.dot(es.T, x), self.mu))) / (n * self.d))
        for k in range(self.k):
            self._update_mu_k(x, binary_latent_factor_approximation, k)
