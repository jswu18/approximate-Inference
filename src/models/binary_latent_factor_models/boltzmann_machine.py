import numpy as np

from src.models.binary_latent_factor_approximations.abstract_binary_latent_factor_approximation import (
    AbstractBinaryLatentFactorApproximation,
)
from src.models.binary_latent_factor_models.binary_latent_factor_model import (
    BinaryLatentFactorModel,
)


class BoltzmannMachine(BinaryLatentFactorModel):
    def __init__(
        self,
        mu: np.ndarray,
        sigma: float,
        pi: np.ndarray,
    ):
        """
        Binary latent factor model with Boltzmann Machine terms
        """
        super().__init__(mu, sigma, pi)

    @property
    def w_matrix(self) -> np.ndarray:
        """
        Weight matrix of the Boltzmann machine

        :return: matrix of weights (number_of_latent_variables, number_of_latent_variables)
        """
        return -self.precision * (self.mu.T @ self.mu)

    def w_matrix_index(self, i, j) -> float:
        """
        Weight matrix at a specific index

        :param i: row index
        :param j: column index
        :return: weight value
        """
        return -self.precision * (self.mu[:, i] @ self.mu[:, j])

    def b(self, x) -> np.ndarray:
        """
        b term in the Boltzmann machine for all data points

        :param x: design matrix (number_of_points, number_of_dimensions)
        :return: matrix of shape (number_of_points, number_of_latent_variables)
        """
        return -(
            self.precision * x @ self.mu
            + self.log_pi_ratio
            - 0.5 * self.precision * np.multiply(self.mu, self.mu).sum(axis=0)
        )

    def b_index(self, x, node_index) -> float:
        """
        b term for a specific node in the Boltzmann machine for all data points

        :param x: design matrix (number_of_points, number_of_dimensions)
        :param node_index: node index
        :return: vector of shape (number_of_points, 1)
        """
        return -(
            self.precision * x @ self.mu[:, node_index]
            + (self.log_pi[0, node_index] - self.log_one_minus_pi[0, node_index])
            - 0.5 * self.precision * self.mu[:, node_index] @ self.mu[:, node_index]
        ).reshape(
            -1,
        )

    @property
    def log_pi_ratio(self) -> np.ndarray:
        return self.log_pi - self.log_one_minus_pi


def init_boltzmann_machine(
    x: np.ndarray,
    binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
) -> BinaryLatentFactorModel:
    """
    Initialise by running a maximisation step with the parameters of the binary latent factor approximation

    :param x: data matrix (number_of_points, number_of_dimensions)
    :param binary_latent_factor_approximation: a binary_latent_factor_approximation
    :return: an initialised Boltzmann machine model
    """
    mu, sigma, pi = BinaryLatentFactorModel.calculate_maximisation_parameters(
        x, binary_latent_factor_approximation
    )
    return BoltzmannMachine(mu=mu, sigma=sigma, pi=pi)
