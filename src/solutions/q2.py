from dataclasses import asdict, fields
from decimal import Decimal

import dataframe_image as dfi
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import scipy

from src.models.bayesian_linear_regression import (
    LinearRegressionParameters,
    Theta,
    compute_linear_regression_posterior,
)
from src.models.gaussian_process_regression import (
    GaussianProcess,
    GaussianProcessParameters,
)
from src.models.kernels import CombinedKernel, CombinedKernelParameters

jax.config.update("jax_enable_x64", True)


def construct_design_matrix(t: np.ndarray):
    return np.stack((t, np.ones(t.shape)), axis=1).T


def a(
    t: np.ndarray,
    y: np.ndarray,
    sigma: float,
    prior_linear_regression_parameters: LinearRegressionParameters,
    save_path: str,
) -> LinearRegressionParameters:
    x = construct_design_matrix(t)
    prior_theta = Theta(
        linear_regression_parameters=prior_linear_regression_parameters,
        sigma=sigma,
    )
    posterior_linear_regression_parameters = compute_linear_regression_posterior(
        x,
        y,
        prior_linear_regression_parameters,
        residuals_precision=prior_theta.precision,
    )
    df_mean = pd.DataFrame(
        posterior_linear_regression_parameters.mean, columns=["value"]
    ).apply(lambda col: ["%.2E" % Decimal(val) for val in col])
    df_mean.index = ["a", "b"]
    df_mean = pd.concat([df_mean], keys=["parameters"])
    dfi.export(df_mean, save_path + "-mean.png")

    df_covariance = pd.DataFrame(
        posterior_linear_regression_parameters.covariance, columns=["a", "b"]
    ).apply(lambda col: ["%.2E" % Decimal(val) for val in col])
    df_covariance.index = ["a", "b"]
    df_covariance = pd.concat([df_covariance], keys=["parameters"])
    df_covariance = pd.concat([df_covariance.T], keys=["parameters"])
    dfi.export(df_covariance, save_path + "-covariance.png")
    return posterior_linear_regression_parameters


def b(
    t_year: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    linear_regression_parameters: LinearRegressionParameters,
    error_mean: float,
    error_variance: float,
    save_path,
) -> None:
    x = construct_design_matrix(t)
    residuals = y - linear_regression_parameters.predict(x)
    plt.plot(t_year.reshape(-1), residuals.reshape(-1))
    plt.xlabel("date (decimal year)")
    plt.ylabel("residual")
    plt.title("2b: g_obs(t)")
    plt.savefig(save_path + "-residuals-timeseries")
    plt.close()

    count, bins = np.histogram(residuals, bins=100, density=True)
    plt.bar(bins[1:], count, label="residuals")
    plt.plot(
        bins[1:],
        scipy.stats.norm.pdf(bins[1:], loc=error_mean, scale=error_variance),
        color="red",
        label="e(t)",
    )
    plt.xlabel("residual bin")
    plt.ylabel("density")
    plt.title("2b: Residuals Density")
    plt.legend()
    plt.savefig(save_path + "-residuals-density-estimation")
    plt.close()


def c(
    kernel: CombinedKernel,
    kernel_parameters: CombinedKernelParameters,
    log_theta_range: np.ndarray,
    t: np.ndarray,
    number_of_samples: int,
    save_path: str,
) -> None:
    gram = kernel(t, **asdict(kernel_parameters))
    plt.imshow(gram)
    plt.xlabel("t")
    plt.ylabel("t")
    plt.title("Gram Matrix (Prior)")
    plt.savefig(save_path + "-gram-matrix")
    plt.close()

    for _ in range(number_of_samples):
        plt.plot(
            np.random.multivariate_normal(
                jnp.zeros(gram.shape[0]), gram, size=1
            ).reshape(-1)
        )
    plt.xlabel("t")
    plt.ylabel("f_GP(t)")
    plt.title("Samples from Gaussian Process Prior")
    plt.savefig(save_path + "-samples")
    plt.close()

    fig_samples, ax_samples = plt.subplots(
        len(fields(kernel_parameters.__class__)),
        len(log_theta_range),
        figsize=(
            len(log_theta_range) * 2,
            len(fields(kernel_parameters.__class__)) * 2,
        ),
        frameon=False,
    )
    for i, field in enumerate(fields(kernel_parameters.__class__)):
        default_value = getattr(kernel_parameters, field.name)
        for j, log_value in enumerate(log_theta_range):
            setattr(kernel_parameters, field.name, log_value)
            gram = kernel(t, **asdict(kernel_parameters))
            ax_samples[i][j].plot(
                np.random.multivariate_normal(
                    jnp.zeros(gram.shape[0]), gram, size=1
                ).reshape(-1),
            )
            ax_samples[i][j].set_title(
                f"{field.name.strip('log_')}={np.round(np.exp(log_value), 2)}"
            )
        setattr(kernel_parameters, field.name, default_value)
    plt.tight_layout()
    plt.savefig(save_path + f"-parameter-samples", bbox_inches="tight")
    plt.close(fig_samples)

    fig_gram, ax_gram = plt.subplots(
        len(fields(kernel_parameters.__class__)),
        len(log_theta_range),
        figsize=(
            len(log_theta_range) * 2,
            len(fields(kernel_parameters.__class__)) * 2,
        ),
        frameon=False,
    )
    for i, field in enumerate(fields(kernel_parameters.__class__)):
        default_value = getattr(kernel_parameters, field.name)
        for j, log_value in enumerate(log_theta_range):
            setattr(kernel_parameters, field.name, log_value)
            gram = kernel(t, **asdict(kernel_parameters))
            ax_gram[i][j].imshow(gram)
            ax_gram[i][j].set_title(
                f"{field.name.strip('log_')}={np.round(np.exp(log_value), 2)}"
            )
        setattr(kernel_parameters, field.name, default_value)
    plt.tight_layout()
    plt.savefig(save_path + f"-parameter-grams", bbox_inches="tight")
    plt.close(fig_gram)


def f(
    t_train: np.ndarray,
    y_train: np.ndarray,
    t_test: np.ndarray,
    min_year: float,
    prior_linear_regression_parameters: LinearRegressionParameters,
    linear_regression_sigma: float,
    kernel: CombinedKernel,
    gaussian_process_parameters: GaussianProcessParameters,
    learning_rate: float,
    number_of_iterations: int,
    save_path: str,
) -> None:
    # Train Bayesian Linear Regression
    x_train = construct_design_matrix(t_train)
    prior_theta = Theta(
        linear_regression_parameters=prior_linear_regression_parameters,
        sigma=linear_regression_sigma,
    )
    posterior_linear_regression_parameters = compute_linear_regression_posterior(
        x_train,
        y_train,
        prior_linear_regression_parameters,
        residuals_precision=prior_theta.precision,
    )

    residuals = y_train - posterior_linear_regression_parameters.predict(x_train)
    gaussian_process = GaussianProcess(
        kernel, t_train.reshape(-1, 1), residuals.reshape(-1)
    )

    # Prediction
    x_test = construct_design_matrix(t_test)
    linear_prediction = posterior_linear_regression_parameters.predict(x_test).reshape(
        -1
    )
    mean_prediction, covariance_prediction = gaussian_process.posterior_distribution(
        t_test.reshape(-1, 1), **asdict(gaussian_process_parameters)
    )

    # Plot
    plt.figure(figsize=(7, 7))
    plt.scatter(
        t_train + min_year,
        y_train.reshape(-1),
        s=2,
        color="blue",
        label="historical data",
    )
    plt.plot(
        t_test + min_year,
        linear_prediction + mean_prediction,
        color="gray",
        label="prediction",
    )
    plt.fill_between(
        t_test + min_year,
        linear_prediction
        + mean_prediction
        - 1 * jnp.sqrt(jnp.diagonal(covariance_prediction)),
        linear_prediction
        + mean_prediction
        + 1 * jnp.sqrt(jnp.diagonal(covariance_prediction)),
        facecolor=(0.8, 0.8, 0.8),
        label="error bound (one stdev)",
    )
    plt.xlabel("date (decimal year)")
    plt.ylabel("parts per million")
    plt.title("Global Mean CO_2 Concentration Prediction (Untrained Hyperparameters)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "-extrapolation-untrained", bbox_inches="tight")
    plt.close()

    df_parameters = pd.DataFrame(
        [
            [
                x.strip("log_") + " (kernel)",
                "%.2E" % Decimal(np.exp(gaussian_process_parameters.kernel[x])),
            ]
            for x in gaussian_process_parameters.kernel.keys()
        ]
        + [["sigma", "%.2E" % Decimal(float(gaussian_process_parameters.sigma))]],
        columns=["parameter", "value"],
    )
    df_parameters = df_parameters.set_index("parameter").sort_values(by=["parameter"])
    dfi.export(df_parameters, save_path + "-untrained-parameters.png")

    # Train Gaussian Process Regression (Hyperparameter Tune)
    optimizer = optax.adam(learning_rate)
    gaussian_process_parameters = gaussian_process.train(
        optimizer, number_of_iterations, **asdict(gaussian_process_parameters)
    )
    df_parameters = pd.DataFrame(
        [
            [
                x.strip("log_") + " (kernel)",
                "%.2E" % Decimal(np.exp(gaussian_process_parameters.kernel[x])),
            ]
            for x in gaussian_process_parameters.kernel.keys()
        ]
        + [["sigma", "%.2E" % Decimal(float(gaussian_process_parameters.sigma))]],
        columns=["parameter", "value"],
    )
    df_parameters = df_parameters.set_index("parameter").sort_values(by=["parameter"])
    dfi.export(df_parameters, save_path + "-trained-parameters.png")

    # Prediction
    x_test = construct_design_matrix(t_test)
    linear_prediction = posterior_linear_regression_parameters.predict(x_test).reshape(
        -1
    )
    mean_prediction, covariance_prediction = gaussian_process.posterior_distribution(
        t_test.reshape(-1, 1), **asdict(gaussian_process_parameters)
    )

    # Plot
    plt.figure(figsize=(7, 7))
    plt.scatter(
        t_train + min_year,
        y_train.reshape(-1),
        s=2,
        color="blue",
        label="historical data",
    )
    plt.plot(
        t_test + min_year,
        linear_prediction + mean_prediction,
        color="gray",
        label="prediction",
    )
    plt.fill_between(
        t_test + min_year,
        linear_prediction
        + mean_prediction
        - 1 * jnp.sqrt(jnp.diagonal(covariance_prediction)),
        linear_prediction
        + mean_prediction
        + 1 * jnp.sqrt(jnp.diagonal(covariance_prediction)),
        facecolor=(0.8, 0.8, 0.8),
        label="error bound (one stdev)",
    )
    plt.xlabel("date (decimal year)")
    plt.ylabel("parts per million")
    plt.title("Global Mean CO_2 Concentration Prediction (Trained Hyperparameters)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + "-extrapolation-trained", bbox_inches="tight")
    plt.close()
