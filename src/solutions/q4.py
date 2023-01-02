import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

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


def offset_image(coord, path, ax):
    img = plt.imread(path)
    im = OffsetImage(img, zoom=0.72)
    im.image.axes = ax

    ab = AnnotationBbox(
        im,
        (coord, 0),
        xybox=(0.0, -19.0),
        frameon=False,
        xycoords="data",
        boxcoords="offset points",
        pad=0,
    )

    ax.add_artist(ab)


def _run_automatic_relevance_determination(
    x: np.ndarray,
    a_parameter: int,
    b_parameter: int,
    k: int,
    em_iterations: int,
    e_maximum_steps: int,
    e_convergence_criterion: float,
) -> Tuple[VariationalBayesBinaryLatentFactorModel, List[float]]:
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
    (_, binary_latent_factor_model, free_energy,) = learn_binary_factors(
        x=x,
        em_iterations=em_iterations,
        binary_latent_factor_model=binary_latent_factor_model,
        binary_latent_factor_approximation=mean_field_approximation,
    )
    return binary_latent_factor_model, free_energy


def b(
    x: np.ndarray,
    a_parameter: int,
    b_parameter: int,
    ks: List[int],
    max_k: int,
    em_iterations: int,
    e_maximum_steps: int,
    e_convergence_criterion: float,
    save_path: str,
) -> None:

    binary_latent_factor_models = []
    free_energies = []
    for i, k in enumerate(ks):
        (
            binary_latent_factor_model,
            free_energy,
        ) = _run_automatic_relevance_determination(
            x,
            a_parameter,
            b_parameter,
            k,
            em_iterations,
            e_maximum_steps,
            e_convergence_criterion,
        )
        binary_latent_factor_models.append(binary_latent_factor_model)
        free_energies.append(free_energy)

    n = len(ks)
    m = np.max(ks)
    fig = plt.figure()
    fig.set_figwidth(2 * n)
    fig.set_figheight(2 * m)
    for i, k in enumerate(ks):
        sort_indices = np.argsort(binary_latent_factor_models[i].gaussian_prior.alpha)
        for j, idx in enumerate(sort_indices):
            ax = plt.subplot(n, m, m * i + j + 1)
            ax.imshow(binary_latent_factor_models[i].mu[:, idx].reshape(4, 4))
            ax.set_title(f"Latent Feature {idx+1}/{k}")
    fig.suptitle("Learned Features (Variational Bayes)")
    plt.tight_layout()
    plt.savefig(save_path + "-latent-factors", bbox_inches="tight")
    plt.close()

    for i, k in enumerate(ks):
        sort_indices = np.argsort(binary_latent_factor_models[i].gaussian_prior.alpha)
        for j, idx in enumerate(sort_indices):
            fig = plt.figure(figsize=(0.3, 0.3))
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(binary_latent_factor_models[i].mu[:, idx].reshape(4, 4))
            fig.savefig(save_path + f"-latent-factor-{i}-{j}", bbox_inches="tight")
            plt.close()

    fig, ax = plt.subplots(len(ks), 1, figsize=(12, 2 * len(ks)))
    plt.subplots_adjust(hspace=1)
    for i, k in enumerate(ks):
        sort_indices = np.argsort(binary_latent_factor_models[i].gaussian_prior.alpha)
        y = list(
            1 / binary_latent_factor_models[i].gaussian_prior.alpha[sort_indices]
        ) + [0] * (max_k - k)
        ax[i].set_title(f"{k=}")
        ax[i].bar(range(max_k), y)
        ax[i].set_xticks([])
        ax[i].set_ylabel("Inverse Alpha")
    for i, k in enumerate(ks):
        sort_indices = np.argsort(binary_latent_factor_models[i].gaussian_prior.alpha)
        for j in range(len(sort_indices)):
            path = save_path + f"-latent-factor-{i}-{j}.png"
            offset_image(j, path, ax[i])
            os.remove(path)
    fig.savefig(save_path + f"-latent-factors-comparison", bbox_inches="tight")
    plt.close()

    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    shades = np.flip(np.linspace(0, 0.7, len(ks)))
    for i, k in enumerate(ks):
        plt.plot(free_energies[i], label=f"{k=}", color=np.ones(3) * shades[i])
    plt.title("Free Energy (Variational Bayes)")
    plt.xlabel("t (EM steps)")
    plt.ylabel("Free Energy")
    plt.legend()
    plt.savefig(save_path + "-free-energy", bbox_inches="tight")
    plt.close()
