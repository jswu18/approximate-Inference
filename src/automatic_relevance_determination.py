from typing import List, Tuple

import numpy as np

from src.expectation_maximisation import learn_binary_factors
from src.models.binary_latent_factor_approximations.mean_field_approximation import (
    init_mean_field_approximation,
)
from src.models.binary_latent_factor_models.binary_latent_factor_model import (
    BinaryLatentFactorModel,
)
from src.models.binary_latent_factor_models.variational_bayes import (
    GaussianPrior,
    VariationalBayesBinaryLatentFactorModel,
)


def run_automatic_relevance_determination(
    x: np.ndarray,
    a_parameter: int,
    b_parameter: int,
    k: int,
    em_iterations: int,
    e_maximum_steps: int,
    e_convergence_criterion: float,
) -> Tuple[VariationalBayesBinaryLatentFactorModel, List[float]]:
    """
    Run automatic relevance determination with variational Bayes.

    :param x: data matrix (number_of_points, number_of_dimensions)
    :param a_parameter: alpha parameter for gamma prior
    :param b_parameter: beta parameter for gamma prior
    :param k: number of latent variables
    :param em_iterations: number of iterations to run EM
    :param e_maximum_steps: maximum number of iterations of partial expectation steps
    :param e_convergence_criterion: minimum required change in free energy for each partial expectation step
    :return: a Tuple containing the optimised VB model and a list of free energies during each EM step
    """
    n = x.shape[0]
    mean_field_approximation = init_mean_field_approximation(
        k, n, max_steps=e_maximum_steps, convergence_criterion=e_convergence_criterion
    )
    (_, sigma, pi,) = BinaryLatentFactorModel.calculate_maximisation_parameters(
        x, mean_field_approximation
    )
    mu = GaussianPrior(
        a=a_parameter,
        b=b_parameter,
        k=k,
        d=x.shape[1],
    )
    binary_latent_factor_model: VariationalBayesBinaryLatentFactorModel = (
        VariationalBayesBinaryLatentFactorModel(
            mu=mu,
            variance=sigma**2,
            pi=pi,
        )
    )
    _, binary_latent_factor_model, free_energy = learn_binary_factors(
        x=x,
        em_iterations=em_iterations,
        binary_latent_factor_model=binary_latent_factor_model,
        binary_latent_factor_approximation=mean_field_approximation,
    )
    return binary_latent_factor_model, free_energy
