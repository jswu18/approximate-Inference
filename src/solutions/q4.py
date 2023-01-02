import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src.automatic_relevance_determination import run_automatic_relevance_determination


def _offset_image(coord: int, path: str, ax: plt.axis):
    """
    Add image to matplotlib axis.

    :param coord: coordinate on axis
    :param path: path to image
    :param ax: plot axis
    """
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
    # automatic relevance determination
    for i, k in enumerate(ks):
        (
            binary_latent_factor_model,
            free_energy,
        ) = run_automatic_relevance_determination(
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

    # store each feature as an image for later use
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

    # bar plot of alphas
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
    # add feature image ticks
    for i, k in enumerate(ks):
        sort_indices = np.argsort(binary_latent_factor_models[i].gaussian_prior.alpha)
        for j in range(len(sort_indices)):
            path = save_path + f"-latent-factor-{i}-{j}.png"
            _offset_image(j, path, ax[i])
            os.remove(path)
    fig.savefig(save_path + f"-latent-factors-comparison", bbox_inches="tight")
    plt.close()

    # free energy plot
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    shades = np.flip(np.linspace(0.3, 0.9, len(ks)))
    for i, k in enumerate(ks):
        plt.plot(free_energies[i], label=f"{k=}", color=np.ones(3) * shades[i])
    plt.title("Free Energy (Variational Bayes)")
    plt.xlabel("t (EM steps)")
    plt.ylabel("Free Energy")
    plt.legend()
    plt.savefig(save_path + "-free-energy", bbox_inches="tight")
    plt.close()
