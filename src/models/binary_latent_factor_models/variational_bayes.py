import numpy as np

from src.models.binary_latent_factor_approximations.abstract_binary_latent_factor_approximation import (
    AbstractBinaryLatentFactorApproximation,
)
from src.models.binary_latent_factor_models.binary_latent_factor_model import (
    BinaryLatentFactorModel,
)
from src.models.binary_latent_factor_models.boltzmann_machine import BoltzmannMachine


class GaussianPrior:
    def __init__(self, a: float, b: float, d: int, k: int):
        """
        Gaussian prior on mu matrix

        :param a: alpha parameter of Gamma Prior
        :param b: beta parameter of Gamma Prior
        :param d: number of dimensions
        :param k: number of latent variables
        """
        self.a = a
        self.b = b
        self.mu = np.zeros((d, k))
        self.alpha = np.ones((k,))
        self.w_covariance = np.zeros((k, k))

    def mu_k(self, k: int) -> np.ndarray:
        """
        Column vector of mu matrix, the latent feature vector

        :param k: latent factor index
        :return: column vector (number_of_dimensions,  1)
        """
        return self.mu[:, k : k + 1]

    def w_d(self, d: int) -> np.ndarray:
        """
        Row vector of mu matrix, the weight vector for a particular dimension (pixel) of the data

        :param d: data dimension index
        :return: row vector (1,  number_of_latent_variables)
        """
        return self.mu[d : d + 1, :]

    @property
    def a_matrix(self) -> np.ndarray:
        """
        Precision matrix for a weight vector w_d
        :return: matrix of shape (number_of_latent_variables, number_of_latent_variables)
        """
        return np.diag(self.alpha)


class VariationalBayes(BoltzmannMachine):
    def __init__(self, mu: GaussianPrior, variance: float, pi: np.ndarray):
        """
        Variational Bayes implementation with prior on mu.
        Note that we are inheriting from BoltzmannMachine for Question 5d only.

        :param mu: Gaussian prior on latent features
        :param variance: Gaussian noise parameter
        :param pi: vector of priors (1, number_of_latent_variables)
        """
        super().__init__(mu=mu.mu, sigma=np.sqrt(variance), pi=pi)
        self.gaussian_prior = mu
        self._variance = variance
        self._pi = pi

    @property
    def variance(self) -> float:
        return self._variance

    @property
    def pi(self) -> np.ndarray:
        return self._pi

    @property
    def mu(self) -> np.ndarray:
        return self.gaussian_prior.mu

    def _update_w_d_covariance(
        self,
        binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
    ):
        #  (number_of_latent_variables, number_of_latent_variables)
        self.gaussian_prior.w_covariance = np.linalg.inv(
            self.gaussian_prior.a_matrix
            + self.precision * binary_latent_factor_approximation.expectation_ss
        )

    def _update_w_d_mean(
        self,
        x: np.ndarray,  # (number_of_points, number_of_dimensions)
        binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
        d: int,
    ) -> None:
        """
        Update mean vector for w_d.

        :param x: data matrix (number_of_points, number_of_dimensions)
        :param binary_latent_factor_approximation: a binary_latent_factor_approximation
        :param d: index of data dimension to update
        :return:
        """

        # (number_of_latent_variables,  1)
        self.gaussian_prior.mu[d : d + 1, :] = (
            self.gaussian_prior.w_covariance
            @ (
                self.precision
                * binary_latent_factor_approximation.expectation_s.T
                @ x[:, d : d + 1]
            )
        ).T

    def _hyper_maximisation_step(self) -> None:
        """
        Hyper M step updating alpha, which parameterize the covariance matrix of the Gaussian prior on mu
        """
        for k in range(self.k):
            self.gaussian_prior.alpha[k] = (2 * self.gaussian_prior.a + self.d - 2) / (
                2 * self.gaussian_prior.b
                + np.sum(self.gaussian_prior.mu_k(k) ** 2)
                + self.d * self.gaussian_prior.w_covariance[k, k]
            )

    def maximisation_step(
        self,
        x: np.ndarray,
        binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
    ) -> None:
        """
        Maximisation step which runs the usual M-step followed by posterior updates to the
        distribution of mu as well as a hyper M-step updating the prior parameters on mu, the alpha vector
        :param x: data matrix (number_of_points, number_of_dimensions)
        :param binary_latent_factor_approximation: a binary_latent_factor_approximation
        """
        _, sigma, pi = BinaryLatentFactorModel.calculate_maximisation_parameters(
            x, binary_latent_factor_approximation
        )
        self._variance = sigma**2
        self._pi = pi
        self._update_w_d_covariance(binary_latent_factor_approximation)
        for d in range(self.d):
            self._update_w_d_mean(x, binary_latent_factor_approximation, d)
        self._hyper_maximisation_step()
