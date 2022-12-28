import numpy as np
from src.models.mean_field_learning import learn_binary_factors, BinaryLatentFactorModel, compute_free_energy, init_mean_field_approximation, variational_expectation_step, is_converge
from src.generate_images import generate_images
import matplotlib.pyplot as plt
from typing import List


def e_and_f(
    x: np.ndarray,
    k: int,
    em_iterations: int,
    e_maximum_steps: int,
    e_convergence_criterion: float,
    save_path: str,
):
    _, binary_latent_factor_model, free_energy = learn_binary_factors(
        x, k, em_iterations, e_maximum_steps, e_convergence_criterion
    )
    fig, ax = plt.subplots(1, k, figsize=(k * 2, 2))
    for i in range(k):
        ax[i].imshow(binary_latent_factor_model.mu[:, i].reshape(4, 4))
        ax[i].set_title(f"Latent Feature mu_{i}")
    fig.suptitle("Learned Features")
    plt.tight_layout()
    plt.savefig(save_path + "-latent-factors", bbox_inches="tight")
    plt.close()

    plt.title("Free Energy")
    plt.xlabel("t (EM steps)")
    plt.ylabel("Free Energy")
    plt.plot(free_energy)
    plt.savefig(save_path + "-free-energy", bbox_inches="tight")
    plt.close()
    return binary_latent_factor_model


def g(
    x: np.ndarray,
    binary_latent_factor_model: BinaryLatentFactorModel,
    sigmas: List[float],
    k: int,
    em_iterations: int,
    e_maximum_steps: int,
    e_convergence_criterion: float,
    save_path: str,
):
    n = x.shape[0]
    free_energies = []
    for sigma in sigmas:
        binary_latent_factor_model.sigma = sigma
        mean_field_approximation = init_mean_field_approximation(k, n)
        free_energy: List[float] = [
            compute_free_energy(x, binary_latent_factor_model, mean_field_approximation)
        ]
        for _ in range(em_iterations):
            new_mean_field_approximation, new_free_energy = variational_expectation_step(
                x=x,
                binary_latent_factor_model=binary_latent_factor_model,
                mean_field_approximation=mean_field_approximation,
                max_steps=e_maximum_steps,
                convergence_criterion=e_convergence_criterion,
            )
            free_energy.extend(new_free_energy)
            if is_converge(
                free_energy, new_mean_field_approximation, mean_field_approximation
            ):
                break
            mean_field_approximation = new_mean_field_approximation
        free_energies.append(free_energy)

    for i, free_energy in enumerate(free_energies):
        plt.plot(np.arange(len(free_energy)-1), np.log(np.diff(np.array(free_energy))), label=f"sigma={sigmas[i]}")
    plt.title(f"log(F(t)-F(t-1)")
    plt.xlabel("t (Variational E steps)")
    plt.ylabel("log(F(t)-F(t-1)")
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path + f"-free-energy-diff-sigma.png", bbox_inches="tight")
    plt.close()
