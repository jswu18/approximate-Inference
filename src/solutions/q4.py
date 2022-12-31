import matplotlib.pyplot as plt
import numpy as np

from src.models.binary_latent_factor_model import (
    BinaryLatentFactorModel,
    learn_binary_factors,
)
from src.models.mean_field_approximation import init_mean_field_approximation
from src.models.variational_bayes import (
    Beta,
    Gaussian,
    InverseGamma,
    VariationalBayesBinaryLatentFactorModel,
)


def b(
    x: np.ndarray,
    k: int,
    em_iterations: int,
    e_maximum_steps: int,
    e_convergence_criterion: float,
    # mu: Gaussian,
    # variance: InverseGamma,
    # pi: Beta,
    save_path: str,
) -> None:
    n = x.shape[0]
    mean_field_approximation = init_mean_field_approximation(
        k, n, max_steps=e_maximum_steps, convergence_criterion=e_convergence_criterion
    )
    (
        mu_max_step,
        sigma,
        pi_max_step,
    ) = BinaryLatentFactorModel.calculate_maximisation_parameters(
        x, mean_field_approximation
    )
    mu = Gaussian(
        mu=mu_max_step,
        variance=mu_max_step.std(axis=0) ** 2,
    )
    variance = InverseGamma(
        a=2,
        b=sigma**2,
    )
    pi = Beta(
        alpha=np.ones((1, k)),
        beta=np.divide(1 - pi_max_step, pi_max_step),
    )
    binary_latent_factor_model = VariationalBayesBinaryLatentFactorModel(
        mu=mu,
        variance=variance,
        pi=pi,
    )
    fig, ax = plt.subplots(1, k, figsize=(k * 2, 2))
    for i in range(k):
        ax[i].imshow(binary_latent_factor_model.mu[:, i].reshape(4, 4))
        ax[i].set_title(f"Latent Feature mu_{i}")
    fig.suptitle("Initial Features (Variational Bayes)")
    plt.tight_layout()
    plt.savefig(save_path + "-init-latent-factors", bbox_inches="tight")
    plt.close()
    (
        mean_field_approximation,
        binary_latent_factor_model,
        free_energy,
    ) = learn_binary_factors(
        x=x,
        em_iterations=em_iterations,
        binary_latent_factor_model=binary_latent_factor_model,
        binary_latent_factor_approximation=mean_field_approximation,
    )
    fig, ax = plt.subplots(1, k, figsize=(k * 2, 2))
    for i in range(k):
        ax[i].imshow(binary_latent_factor_model.mu[:, i].reshape(4, 4))
        ax[i].set_title(f"Latent Feature mu_{i}")
    fig.suptitle("Learned Features (Variational Bayes)")
    plt.tight_layout()
    plt.savefig(save_path + "-latent-factors", bbox_inches="tight")
    plt.close()

    plt.title("Free Energy (Variational Bayes)")
    plt.xlabel("t (EM steps)")
    plt.ylabel("Free Energy")
    plt.plot(free_energy)
    plt.savefig(save_path + "-free-energy", bbox_inches="tight")
    plt.close()
