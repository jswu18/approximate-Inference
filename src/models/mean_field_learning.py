from dataclasses import dataclass

import numpy as np

from demo_code.MStep import m_step


@dataclass
class MeanFieldApproximation:
    """
    lambda_matrix: parameters variational approximation (number_of_points, number_of_latent_variables)
    """

    lambda_matrix: np.ndarray

    def lambda_matrix_exclude(self, exclude_latent_index: int) -> np.ndarray:
        #  (number_of_points, number_of_latent_variables-1)
        return np.concatenate(
            (
                self.lambda_matrix[:, :exclude_latent_index],
                self.lambda_matrix[:, exclude_latent_index + 1 :],
            ),
            axis=1,
        )

    @property
    def log_lambda_matrix(self):
        return np.log(self.lambda_matrix)

    @property
    def log_one_minus_lambda_matrix(self):
        return np.log(1 - self.lambda_matrix)

    @property
    def n(self):
        return self.lambda_matrix.shape[0]

    @property
    def k(self):
        return self.lambda_matrix.shape[1]


def init_mean_field_approximation(k: int, n: int) -> MeanFieldApproximation:
    return MeanFieldApproximation(
        lambda_matrix=np.random.random(size=(n, k)),
    )


@dataclass
class BinaryLatentFactorModel:
    """
    mu: matrix of means (number_of_dimensions, number_of_latent_variables)
    sigma: gaussian noise parameter
    pi: vector of priors (1, number_of_latent_variables)
    """

    mu: np.ndarray
    sigma: float
    pi: np.ndarray

    def mu_exclude(self, exclude_latent_index: int) -> np.ndarray:
        #  (number_of_dimensions, number_of_latent_variables-1)
        return np.concatenate(
            (self.mu[:, :exclude_latent_index], self.mu[:, exclude_latent_index + 1 :]),
            axis=1,
        )

    @property
    def log_pi(self):
        return np.log(self.pi)

    @property
    def log_one_minus_pi(self):
        return np.log(1 - self.pi)

    @property
    def variance(self):
        return self.sigma**2

    @property
    def precision(self):
        return 1 / self.variance

    @property
    def d(self):
        return self.mu.shape[0]

    @property
    def k(self):
        return self.mu.shape[1]


def init_binary_latent_factor_model(
    x: np.ndarray,
    mean_field_approximation: MeanFieldApproximation,
) -> BinaryLatentFactorModel:
    return maximisation_step(x, mean_field_approximation)


def _compute_expectation_log_p_x_s_given_theta(
    x: np.ndarray,
    binary_latent_factor_model: BinaryLatentFactorModel,
    mean_field_approximation: MeanFieldApproximation,
) -> float:
    """
    The first term of the free energy, the expectation of log P(X,S|theta)

    :param x: data matrix (number_of_points, number_of_dimensions)
    :param binary_latent_factor_model: a binary_latent_factor_model
    :param mean_field_approximation: a mean_field_approximation
    :return: the expectation of log P(X,S|theta)
    """
    # (number_of_points, number_of_dimensions)
    mu_lambda = mean_field_approximation.lambda_matrix @ binary_latent_factor_model.mu.T

    # (number_of_latent_variables, number_of_latent_variables)
    expectation_s_i_s_j_mu_i_mu_j = np.multiply(
        mean_field_approximation.lambda_matrix.T
        @ mean_field_approximation.lambda_matrix,
        binary_latent_factor_model.mu.T @ binary_latent_factor_model.mu,
    )

    expectation_log_p_x_given_s_theta = -(
        mean_field_approximation.n * binary_latent_factor_model.d / 2
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
            mean_field_approximation.lambda_matrix
            @ np.multiply(
                binary_latent_factor_model.mu, binary_latent_factor_model.mu
            ).T
        )
    )
    expectation_log_p_s_given_theta = np.sum(
        np.multiply(
            mean_field_approximation.lambda_matrix,
            binary_latent_factor_model.log_pi,
        )
        + np.multiply(
            1 - mean_field_approximation.lambda_matrix,
            binary_latent_factor_model.log_one_minus_pi,
        )
    )
    return expectation_log_p_x_given_s_theta + expectation_log_p_s_given_theta


def _compute_mean_field_approximation_entropy(
    mean_field_approximation: MeanFieldApproximation,
) -> float:
    return -np.sum(
        np.multiply(
            mean_field_approximation.lambda_matrix,
            mean_field_approximation.log_lambda_matrix,
        )
        + np.multiply(
            1 - mean_field_approximation.lambda_matrix,
            mean_field_approximation.log_one_minus_lambda_matrix,
        )
    )


def compute_free_energy(
    x: np.ndarray,
    binary_latent_factor_model: BinaryLatentFactorModel,
    mean_field_approximation: MeanFieldApproximation,
) -> float:
    """
    free energy associated with current EM parameters and data x

    :param x: data matrix (number_of_points, number_of_dimensions)
    :param binary_latent_factor_model: a binary_latent_factor_model
    :param mean_field_approximation: a mean_field_approximation
    :return: average free energy per data point
    """
    expectation_log_p_x_s_given_theta = _compute_expectation_log_p_x_s_given_theta(
        x, binary_latent_factor_model, mean_field_approximation
    )
    mean_field_approximation_entropy = _compute_mean_field_approximation_entropy(
        mean_field_approximation
    )
    return (
        expectation_log_p_x_s_given_theta + mean_field_approximation_entropy
    ) / mean_field_approximation.n


def partial_expectation_step(
    x: np.ndarray,
    binary_latent_factor_model: BinaryLatentFactorModel,
    mean_field_approximation: MeanFieldApproximation,
    latent_factor: int,
) -> np.ndarray:
    """Partial Variational E step for factor i for all data points

    :param x: data matrix (number_of_points, number_of_dimensions)
    :param binary_latent_factor_model: a binary_latent_factor_model
    :param mean_field_approximation: a mean_field_approximation
    :param latent_factor: latent factor to compute partial update
    :return: lambda_vector: new lambda parameters for the latent factor (number_of_points, 1)
    """
    lambda_matrix_excluded = mean_field_approximation.lambda_matrix_exclude(
        latent_factor
    )
    mu_excluded = binary_latent_factor_model.mu_exclude(latent_factor)

    mu_latent = binary_latent_factor_model.mu[:, latent_factor]
    #  (number_of_points, 1)
    partial_expectation_log_p_x_given_s_theta_proportion = (
        binary_latent_factor_model.precision
        * (
            x  # (number_of_points, number_of_dimensions)
            - 0.5 * mu_latent.T  # (1, number_of_dimensions)
            - lambda_matrix_excluded  # (number_of_points, number_of_latent_variables-1)
            @ mu_excluded.T  # (number_of_latent_variables-1, number_of_dimensions)
        )
        @ mu_latent  # (number_of_dimensions, 1)
    )

    #  (1, 1)
    partial_expectation_log_p_s_given_theta_proportion = np.log(
        binary_latent_factor_model.pi[0, latent_factor]
        / (1 - binary_latent_factor_model.pi[0, latent_factor])
    )

    #  (number_of_points, 1)
    partial_expectation_log_p_x_s_given_theta_proportion = (
        partial_expectation_log_p_x_given_s_theta_proportion
        + partial_expectation_log_p_s_given_theta_proportion
    )

    #  (number_of_points, 1)
    lambda_vector = 1 / (
        1 + np.exp(-partial_expectation_log_p_x_s_given_theta_proportion)
    )
    lambda_vector[lambda_vector == 0] = 1e-10
    lambda_vector[lambda_vector == 1] = 1 - 1e-10
    return lambda_vector


def variational_expectation_step(
    x: np.ndarray,
    binary_latent_factor_model: BinaryLatentFactorModel,
    mean_field_approximation: MeanFieldApproximation,
    max_steps: int,
    convergence_criterion: float,
) -> MeanFieldApproximation:
    """Variational E step

    :param x: data matrix (number_of_points, number_of_dimensions)
    :param binary_latent_factor_model: a binary_latent_factor_model
    :param mean_field_approximation: a mean_field_approximation
    :param max_steps: maximum number of steps of fixed point equations
    :param convergence_criterion: early stopping if change in free energy <  convergence_criterion
    :return: mean field approximation
    """
    previous_free_energy = compute_free_energy(
        x, binary_latent_factor_model, mean_field_approximation
    )
    for i in range(max_steps):
        for latent_factor in range(binary_latent_factor_model.k):
            mean_field_approximation.lambda_matrix[
                :, latent_factor
            ] = partial_expectation_step(
                x, binary_latent_factor_model, mean_field_approximation, latent_factor
            )
        free_energy = compute_free_energy(
            x, binary_latent_factor_model, mean_field_approximation
        )
        if free_energy - previous_free_energy <= convergence_criterion:
            break
        previous_free_energy = free_energy
    return mean_field_approximation


def maximisation_step(
    x: np.ndarray,
    mean_field_approximation: MeanFieldApproximation,
) -> BinaryLatentFactorModel:
    expectation_s = mean_field_approximation.lambda_matrix
    expectation_ss = (
        mean_field_approximation.lambda_matrix.T
        @ mean_field_approximation.lambda_matrix
    )
    np.fill_diagonal(expectation_ss, mean_field_approximation.lambda_matrix.sum(axis=0))
    mu, sigma, pi = m_step(x, expectation_s, expectation_ss)
    return BinaryLatentFactorModel(
        mu=mu,
        sigma=sigma,
        pi=pi,
    )


def learn_binary_factors(
    x: np.ndarray,
    k: int,
    em_maximum_iterations: int,
    e_maximum_steps: int,
    e_convergence_criterion: float,
):
    n = x.shape[0]
    mean_field_approximation = init_mean_field_approximation(k, n)
    binary_latent_factor_model = init_binary_latent_factor_model(
        x, mean_field_approximation
    )

    for _ in range(em_maximum_iterations):
        mean_field_approximation = variational_expectation_step(
            x=x,
            binary_latent_factor_model=binary_latent_factor_model,
            mean_field_approximation=mean_field_approximation,
            max_steps=e_maximum_steps,
            convergence_criterion=e_convergence_criterion,
        )
        binary_latent_factor_model = maximisation_step(
            x=x,
            mean_field_approximation=mean_field_approximation,
        )
    return mean_field_approximation, binary_latent_factor_model
