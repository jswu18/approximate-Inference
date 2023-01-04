from typing import List, Tuple

import numpy as np

from src.expectation_maximisation import learn_binary_factors
from src.models.binary_latent_factor_approximations.abstract_binary_latent_factor_approximation import (
    AbstractBinaryLatentFactorApproximation,
)
from src.models.binary_latent_factor_models.binary_latent_factor_model import (
    BinaryLatentFactorModel,
)
from src.models.binary_latent_factor_models.variational_bayes import (
    GaussianPrior,
    VariationalBayes,
)


def run_automatic_relevance_determination(
    x: np.ndarray,
    binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
    a_parameter: float,
    b_parameter: float,
    k: int,
    em_iterations: int,
) -> Tuple[VariationalBayes, List[float]]:
    """
    Run automatic relevance determination with variational Bayes.

    :param x: data matrix (number_of_points, number_of_dimensions)
    :param binary_latent_factor_approximation: a binary_latent_factor_approximation
    :param a_parameter: alpha parameter for gamma prior
    :param b_parameter: beta parameter for gamma prior
    :param k: number of latent variables
    :param em_iterations: number of iterations to run EM
    :return: a Tuple containing the optimised VB model and a list of free energies during each EM step
    """
    (_, sigma, pi,) = BinaryLatentFactorModel.calculate_maximisation_parameters(
        x, binary_latent_factor_approximation
    )
    mu = GaussianPrior(
        a=a_parameter,
        b=b_parameter,
        k=k,
        d=x.shape[1],
    )
    variational_bayes_model: VariationalBayes = VariationalBayes(
        mu=mu,
        variance=sigma**2,
        pi=pi,
    )
    _, variational_bayes_model, free_energy = learn_binary_factors(
        x=x,
        em_iterations=em_iterations,
        binary_latent_factor_model=variational_bayes_model,
        binary_latent_factor_approximation=binary_latent_factor_approximation,
    )
    return variational_bayes_model, free_energy
