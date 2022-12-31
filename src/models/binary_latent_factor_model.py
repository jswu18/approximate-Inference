from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from demo_code.MStep import m_step


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


class BinaryLatentFactorApproximation(ABC):
    @property
    @abstractmethod
    def lambda_matrix(self) -> np.ndarray:
        pass

    @abstractmethod
    def variational_expectation_step(
        self,
        x: np.ndarray,
        binary_latent_factor_model: AbstractBinaryLatentFactorModel,
    ) -> List[float]:
        pass

    @property
    def expectation_s(self):
        return self.lambda_matrix

    @property
    def expectation_ss(self):
        ess = self.lambda_matrix.T @ self.lambda_matrix
        np.fill_diagonal(ess, self.lambda_matrix.sum(axis=0))
        return ess

    @property
    def log_lambda_matrix(self) -> np.ndarray:
        return np.log(self.lambda_matrix)

    @property
    def log_one_minus_lambda_matrix(self) -> np.ndarray:
        return np.log(1 - self.lambda_matrix)

    @property
    def n(self) -> int:
        return self.lambda_matrix.shape[0]

    @property
    def k(self) -> int:
        return self.lambda_matrix.shape[1]

    def compute_free_energy(
        self,
        x: np.ndarray,
        binary_latent_factor_model: AbstractBinaryLatentFactorModel,
    ) -> float:
        """
        free energy associated with current EM parameters and data x

        :param x: data matrix (number_of_points, number_of_dimensions)
        :param binary_latent_factor_model: a binary_latent_factor_model
        :return: average free energy per data point
        """
        expectation_log_p_x_s_given_theta = (
            self._compute_expectation_log_p_x_s_given_theta(
                x, binary_latent_factor_model
            )
        )
        approximation_model_entropy = self._compute_approximation_model_entropy()
        return (
            expectation_log_p_x_s_given_theta + approximation_model_entropy
        ) / self.n

    def _compute_expectation_log_p_x_s_given_theta(
        self,
        x: np.ndarray,
        binary_latent_factor_model: AbstractBinaryLatentFactorModel,
    ) -> float:
        """
        The first term of the free energy, the expectation of log P(X,S|theta)

        :param x: data matrix (number_of_points, number_of_dimensions)
        :param binary_latent_factor_model: a binary_latent_factor_model
        :return: the expectation of log P(X,S|theta)
        """
        # (number_of_points, number_of_dimensions)
        mu_lambda = self.lambda_matrix @ binary_latent_factor_model.mu.T

        # (number_of_latent_variables, number_of_latent_variables)
        expectation_s_i_s_j_mu_i_mu_j = np.multiply(
            self.lambda_matrix.T @ self.lambda_matrix,
            binary_latent_factor_model.mu.T @ binary_latent_factor_model.mu,
        )

        expectation_log_p_x_given_s_theta = -(
            self.n * binary_latent_factor_model.d / 2
        ) * np.log(2 * np.pi * binary_latent_factor_model.variance) - (
            0.5 * binary_latent_factor_model.precision
        ) * (
            np.sum(np.multiply(x, x))
            - 2 * np.sum(np.multiply(x, mu_lambda))
            + np.sum(expectation_s_i_s_j_mu_i_mu_j)
            - np.trace(
                expectation_s_i_s_j_mu_i_mu_j
            )  # remove incorrect E[s_i s_i] = lambda_i * lambda_i
            + np.sum(  # add correct E[s_i s_i] = lambda_i
                self.lambda_matrix
                @ np.multiply(
                    binary_latent_factor_model.mu, binary_latent_factor_model.mu
                ).T
            )
        )
        expectation_log_p_s_given_theta = np.sum(
            np.multiply(
                self.lambda_matrix,
                binary_latent_factor_model.log_pi,
            )
            + np.multiply(
                1 - self.lambda_matrix,
                binary_latent_factor_model.log_one_minus_pi,
            )
        )
        return expectation_log_p_x_given_s_theta + expectation_log_p_s_given_theta

    def _compute_approximation_model_entropy(self) -> float:
        return -np.sum(
            np.multiply(
                self.lambda_matrix,
                self.log_lambda_matrix,
            )
            + np.multiply(
                1 - self.lambda_matrix,
                self.log_one_minus_lambda_matrix,
            )
        )


def is_converge(
    free_energies: List[float],
    current_lambda_matrix: np.ndarray,
    previous_lambda_matrix: np.ndarray,
) -> bool:
    return (abs(free_energies[-1] - free_energies[-2]) == 0) and np.linalg.norm(
        current_lambda_matrix - previous_lambda_matrix
    ) == 0


def learn_binary_factors(
    x: np.ndarray,
    em_iterations: int,
    binary_latent_factor_model: AbstractBinaryLatentFactorModel,
    binary_latent_factor_approximation: BinaryLatentFactorApproximation,
) -> Tuple[
    BinaryLatentFactorApproximation, AbstractBinaryLatentFactorModel, List[float]
]:
    free_energies: List[float] = [
        binary_latent_factor_approximation.compute_free_energy(
            x, binary_latent_factor_model
        )
    ]
    for _ in range(em_iterations):
        previous_lambda_matrix = np.copy(
            binary_latent_factor_approximation.lambda_matrix
        )
        binary_latent_factor_approximation.variational_expectation_step(
            x=x,
            binary_latent_factor_model=binary_latent_factor_model,
        )
        binary_latent_factor_model.maximisation_step(
            x,
            binary_latent_factor_approximation,
        )
        free_energies.append(
            binary_latent_factor_approximation.compute_free_energy(
                x, binary_latent_factor_model
            )
        )
        if is_converge(
            free_energies,
            binary_latent_factor_approximation.lambda_matrix,
            previous_lambda_matrix,
        ):
            break
    return binary_latent_factor_approximation, binary_latent_factor_model, free_energies
