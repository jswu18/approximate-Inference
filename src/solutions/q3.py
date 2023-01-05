from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.expectation_maximisation import is_converge, learn_binary_factors
from src.models.binary_latent_factor_approximations.mean_field_approximation import (
    MeanFieldApproximation,
    init_mean_field_approximation,
)
from src.models.binary_latent_factor_models.binary_latent_factor_model import (
    AbstractBinaryLatentFactorModel,
    init_binary_latent_factor_model,
)


def e_and_f(
    x: np.ndarray,
    k: int,
    em_iterations: int,
    e_maximum_steps: int,
    e_convergence_criterion: float,
    save_path: str,
) -> [AbstractBinaryLatentFactorModel, MeanFieldApproximation]:
    n = x.shape[0]
    mean_field_approximation = init_mean_field_approximation(
        k, n, max_steps=e_maximum_steps, convergence_criterion=e_convergence_criterion
    )
    binary_latent_factor_model = init_binary_latent_factor_model(
        x, mean_field_approximation
    )
    fig, ax = plt.subplots(1, k, figsize=(k * 2, 2))
    for i in range(k):
        ax[i].imshow(binary_latent_factor_model.mu[:, i].reshape(4, 4))
        ax[i].set_title(f"Latent Feature mu_{i}")
    fig.suptitle("Initial Features (Mean Field Learning)")
    plt.tight_layout()
    plt.savefig(save_path + "-init-latent-factors", bbox_inches="tight")
    plt.close()
    (
        mean_field_approximation,
        binary_latent_factor_model,
        free_energy,
    ) = learn_binary_factors(
        x,
        em_iterations,
        binary_latent_factor_model,
        binary_latent_factor_approximation=mean_field_approximation,
    )
    fig, ax = plt.subplots(1, k, figsize=(k * 2, 2))
    for i in range(k):
        ax[i].imshow(binary_latent_factor_model.mu[:, i].reshape(4, 4))
        ax[i].set_title(f"Latent Feature mu_{i}")
    fig.suptitle("Learned Features (Mean Field Learning)")
    plt.tight_layout()
    plt.savefig(save_path + "-latent-factors", bbox_inches="tight")
    plt.close()

    plt.title("Free Energy (Mean Field Learning)")
    plt.xlabel("t (EM steps)")
    plt.ylabel("Free Energy")
    plt.plot(free_energy)
    plt.savefig(save_path + "-free-energy", bbox_inches="tight")
    plt.close()
    return binary_latent_factor_model, mean_field_approximation


def g(
    x: np.ndarray,
    binary_latent_factor_model: AbstractBinaryLatentFactorModel,
    mean_field_approximation: MeanFieldApproximation,
    sigmas: List[float],
    em_iterations: int,
    save_path: str,
) -> None:
    free_energies = []
    for sigma in sigmas:
        binary_latent_factor_model.sigma = sigma
        mean_field_approximation_single_point = MeanFieldApproximation(
            lambda_matrix=mean_field_approximation.lambda_matrix[:1, :],
            max_steps=mean_field_approximation.max_steps,
            convergence_criterion=mean_field_approximation.convergence_criterion,
        )
        free_energy: List[float] = [
            mean_field_approximation_single_point.compute_free_energy(
                x, binary_latent_factor_model
            )
        ]
        for _ in range(em_iterations):
            free_energy.pop(-1)
            previous_lambda_matrix = np.copy(
                mean_field_approximation_single_point.lambda_matrix
            )
            new_free_energy = (
                mean_field_approximation_single_point.variational_expectation_step(
                    binary_latent_factor_model=binary_latent_factor_model,
                    x=x,
                )
            )
            free_energy.extend(new_free_energy)
            if (
                free_energy[-1] - free_energy[-2]
                <= mean_field_approximation_single_point.convergence_criterion
            ):
                free_energy.pop(-1)
                break
            if is_converge(
                free_energy,
                mean_field_approximation_single_point.lambda_matrix,
                previous_lambda_matrix,
            ):
                break
        free_energies.append(free_energy)

    for i, free_energy in enumerate(free_energies):
        plt.plot(
            free_energy,
            label=f"sigma={sigmas[i]}",
        )
    plt.title(f"F(t)")
    plt.xlabel("t (Variational E steps)")
    plt.ylabel("F(t)")
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path + f"-free-energy-sigma.png", bbox_inches="tight")
    plt.close()

    for i, free_energy in enumerate(free_energies):
        diffs = np.log(np.diff(free_energy))
        plt.plot(
            diffs,
            label=f"sigma={sigmas[i]}",
        )
    plt.title(f"log(F(t)-F(t-1)")
    plt.xlabel("t (Variational E steps)")
    plt.ylabel("log(F(t)-F(t-1)")
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path + f"-free-energy-diff-sigma.png", bbox_inches="tight")
    plt.close()
