from typing import TYPE_CHECKING, Tuple

import numpy as np

from demo_code.MStep import m_step
from src.models.binary_latent_factor_approximations.abstract_binary_latent_factor_approximation import (
    AbstractBinaryLatentFactorApproximation,
)
from src.models.binary_latent_factor_models.abstract_binary_latent_factor_model import (
    AbstractBinaryLatentFactorModel,
)


class BinaryLatentFactorModel(AbstractBinaryLatentFactorModel):
    def __init__(
        self,
        mu: np.ndarray,
        sigma: float,
        pi: np.ndarray,
    ):
        """

        :param mu: matrix of means (number_of_dimensions, number_of_latent_variables)
        :param sigma: Gaussian noise parameter
        :param pi: vector of priors (1, number_of_latent_variables)
        """
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
        binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        return m_step(
            x=x,
            es=binary_latent_factor_approximation.expectation_s,
            ess=binary_latent_factor_approximation.expectation_ss,
        )

    def maximisation_step(
        self,
        x: np.ndarray,
        binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
    ) -> None:
        mu, sigma, pi = self.calculate_maximisation_parameters(
            x, binary_latent_factor_approximation
        )
        self.mu = mu
        self.sigma = sigma
        self.pi = pi


def init_binary_latent_factor_model(
    x: np.ndarray,
    binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
) -> BinaryLatentFactorModel:
    """
    Initialise binary latent factor model by running a maximisation step with the parameters of the
    binary latent factor approximation

    :param x: data matrix (number_of_points, number_of_dimensions)
    :param binary_latent_factor_approximation: a binary_latent_factor_approximation
    :return: an initialised binary latent factor model
    """
    mu, sigma, pi = BinaryLatentFactorModel.calculate_maximisation_parameters(
        x, binary_latent_factor_approximation
    )
    return BinaryLatentFactorModel(mu, sigma, pi)
