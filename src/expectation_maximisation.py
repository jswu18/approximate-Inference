from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.models.binary_latent_factor_approximations.abstract_binary_latent_factor_approximation import (
    AbstractBinaryLatentFactorApproximation,
)
from src.models.binary_latent_factor_models.binary_latent_factor_model import (
    AbstractBinaryLatentFactorModel,
)


def is_converge(
    free_energies: List[float],
    current_lambda_matrix: np.ndarray,
    previous_lambda_matrix: np.ndarray,
) -> bool:
    """
    Check for convergence of free energy and lambda matrix

    :param free_energies: list of free energies
    :param current_lambda_matrix: current lambda matrix
    :param previous_lambda_matrix: previous lambda matrix
    :return: boolean indicating convergence
    """
    return (abs(free_energies[-1] - free_energies[-2]) == 0) and np.linalg.norm(
        current_lambda_matrix - previous_lambda_matrix
    ) == 0


def learn_binary_factors(
    x: np.ndarray,
    em_iterations: int,
    binary_latent_factor_model: AbstractBinaryLatentFactorModel,
    binary_latent_factor_approximation: AbstractBinaryLatentFactorApproximation,
) -> Tuple[
    AbstractBinaryLatentFactorApproximation,
    AbstractBinaryLatentFactorModel,
    List[float],
]:
    """
    Expectation maximisation algorithm to learn binary factors.

    :param x: data matrix (number_of_points, number_of_dimensions)
    :param em_iterations: number of iterations to run EM
    :param binary_latent_factor_model: a binary_latent_factor_model
    :param binary_latent_factor_approximation: a binary_latent_factor_approximation
    :return: a Tuple containing the updated binary_latent_factor_model, updated binary_latent_factor_approximation,
             and free energies during each step of EM
    """
    free_energies: List[float] = [
        binary_latent_factor_approximation.compute_free_energy(
            x, binary_latent_factor_model
        )
    ]
    for _ in range(em_iterations):
        previous_lambda_matrix = np.copy(
            binary_latent_factor_approximation.lambda_matrix
        )

        # E step
        binary_latent_factor_approximation.variational_expectation_step(
            x=x,
            binary_latent_factor_model=binary_latent_factor_model,
        )

        # M step
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
