import numpy as np

from src.models.binary_latent_factor_approximations.abstract_binary_latent_factor_approximation import (
    AbstractBinaryLatentFactorApproximation,
)
from src.models.binary_latent_factor_models.binary_latent_factor_model import (
    AbstractBinaryLatentFactorModel,
    BinaryLatentFactorModel,
)


class GaussianPrior:
    def __init__(self, a, b, d, k):
        self.a = a
        self.b = b
        self.mu = np.zeros((d, k))
        self.alpha = np.ones((k,))  # np.random.gamma(a, b, size=(k,))
        self.w_covariance = np.zeros((k, k))

    def mu_k(self, k):  # (number_of_dimensions,  1)
        return self.mu[:, k : k + 1]

    def w_d(self, d):  # (1,  number_of_latent_variables)
        return self.mu[d : d + 1, :]

    @property
    def a_matrix(self) -> np.ndarray:
        #  precision matrix for w_d
        return np.diag(self.alpha)


class VariationalBayesBinaryLatentFactorModel(AbstractBinaryLatentFactorModel):
    def __init__(self, mu: GaussianPrior, variance: float, pi: np.ndarray):
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
        #  expectation_s (number_of_points, number_of_latent_variables)
        #  expectation_ss (number_of_latent_variables, number_of_latent_variables)
        self.gaussian_prior.w_covariance = np.linalg.inv(
            self.gaussian_prior.a_matrix
            + self.precision * binary_latent_factor_approximation.expectation_ss
        )

    def _update_w_d_mean(
        self,
        x: np.ndarray,  # (number_of_points, number_of_dimensions)
        binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
        d: int,
    ):
        # (number_of_latent_variables x 1)
        self.gaussian_prior.mu[d : d + 1, :] = (
            self.gaussian_prior.w_covariance
            @ (  # (number_of_latent_variables, number_of_latent_variables)
                self.precision
                * binary_latent_factor_approximation.expectation_s.T  # (number_of_latent_variables, number_of_points)
                @ x[:, d : d + 1]  # (number_of_points, 1)
            )
        ).T

    def _hyper_maximisation_step(self):
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
        _, sigma, pi = BinaryLatentFactorModel.calculate_maximisation_parameters(
            x, binary_latent_factor_approximation
        )
        self._variance = sigma**2
        self._pi = pi
        self._update_w_d_covariance(binary_latent_factor_approximation)
        for d in range(self.d):
            self._update_w_d_mean(x, binary_latent_factor_approximation, d)
        self._hyper_maximisation_step()
