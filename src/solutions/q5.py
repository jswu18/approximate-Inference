from typing import List

import numpy as np
from tqdm import tqdm

from src.automatic_relevance_determination import run_automatic_relevance_determination
from src.models.binary_latent_factor_approximations.message_passing_approximation import (
    init_message_passing,
)
from src.models.binary_latent_factor_models.variational_bayes import VariationalBayes
from src.solutions.q4 import plot_factors


def d(
    x: np.ndarray,
    a_parameter: int,
    b_parameter: int,
    ks: List[int],
    max_k: int,
    em_iterations: int,
    save_path: str,
) -> List[List[float]]:

    variational_bayes_models: List[VariationalBayes] = []
    free_energies = []
    for i, k in tqdm(enumerate(ks)):
        n = x.shape[0]
        message_passing_approximation = init_message_passing(k, n)
        (variational_bayes_model, free_energy) = run_automatic_relevance_determination(
            x=x,
            binary_latent_factor_approximation=message_passing_approximation,
            a_parameter=a_parameter,
            b_parameter=b_parameter,
            k=k,
            em_iterations=em_iterations,
        )
        variational_bayes_models.append(variational_bayes_model)
        free_energies.append(free_energy)
    plot_factors(
        variational_bayes_models,
        ks,
        max_k,
        save_path,
    )
    return free_energies
