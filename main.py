import os
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from src.constants import CO2_FILE_PATH, DEFAULT_SEED, OUTPUTS_FOLDER
from src.generate_images import generate_images
from src.models.bayesian_linear_regression import LinearRegressionParameters
from src.models.gaussian_process_regression import GaussianProcessParameters
from src.models.kernels import CombinedKernel, CombinedKernelParameters
from src.solutions import q2, q3, q4, q6

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    np.random.seed(DEFAULT_SEED)

    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)

    # Question 2
    Q2_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q2")
    if not os.path.exists(Q2_OUTPUT_FOLDER):
        os.makedirs(Q2_OUTPUT_FOLDER)
    with open(CO2_FILE_PATH) as file:
        lines = [line.rstrip().split() for line in file]

    df_co2 = pd.DataFrame(
        np.array([line for line in lines if line[0] != "#"]).astype(float)
    )
    column_names = lines[max([i for i, line in enumerate(lines) if line[0] == "#"])][1:]
    df_co2.columns = column_names
    t = df_co2.decimal.values[:] - np.min(df_co2.decimal.values[:])
    y = df_co2.average.values[:].reshape(1, -1)

    sigma = 1
    mean = np.array([0, 360]).reshape(-1, 1)
    covariance = np.array(
        [
            [10**2, 0],
            [0, 100**2],
        ]
    )
    kernel = CombinedKernel()
    kernel_parameters = CombinedKernelParameters(
        log_theta=jnp.log(1),
        log_sigma=jnp.log(1),
        log_phi=jnp.log(1),
        log_eta=jnp.log(1),
        log_tau=jnp.log(1),
        log_zeta=jnp.log(1e-1),
    )

    prior_linear_regression_parameters = LinearRegressionParameters(
        mean=mean,
        covariance=covariance,
    )
    posterior_linear_regression_parameters = q2.a(
        t,
        y,
        sigma,
        prior_linear_regression_parameters,
        save_path=os.path.join(Q2_OUTPUT_FOLDER, "a"),
    )
    q2.b(
        t_year=df_co2.decimal.values[:],
        t=t,
        y=y,
        linear_regression_parameters=posterior_linear_regression_parameters,
        error_mean=0,
        error_variance=1,
        save_path=os.path.join(Q2_OUTPUT_FOLDER, "b"),
    )

    q2.c(
        kernel=kernel,
        kernel_parameters=kernel_parameters,
        log_theta_range=jnp.log(jnp.linspace(1e-2, 5, 5)),
        t=t[:50].reshape(-1, 1),
        number_of_samples=3,
        save_path=os.path.join(Q2_OUTPUT_FOLDER, "c"),
    )

    init_kernel_parameters = CombinedKernelParameters(
        log_theta=jnp.log(5),
        log_sigma=jnp.log(5),
        log_phi=jnp.log(10),
        log_eta=jnp.log(5),
        log_tau=jnp.log(1),
        log_zeta=jnp.log(2),
    )
    gaussian_process_parameters = GaussianProcessParameters(
        kernel=asdict(init_kernel_parameters),
        log_sigma=jnp.log(1),
    )
    years_to_predict = 15
    t_new = t[-1] + np.linspace(0, years_to_predict, years_to_predict * 12)
    t_test = np.concatenate((t, t_new))
    q2.f(
        t_train=t,
        y_train=y,
        t_test=t_test,
        min_year=np.min(df_co2.decimal.values[:]),
        prior_linear_regression_parameters=prior_linear_regression_parameters,
        linear_regression_sigma=sigma,
        kernel=kernel,
        gaussian_process_parameters=gaussian_process_parameters,
        learning_rate=1e-2,
        number_of_iterations=100,
        save_path=os.path.join(Q2_OUTPUT_FOLDER, "f"),
    )

    # Question 3
    Q3_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q3")
    if not os.path.exists(Q3_OUTPUT_FOLDER):
        os.makedirs(Q3_OUTPUT_FOLDER)
    number_of_images = 2000
    x = generate_images(n=number_of_images)
    k = 8
    em_iterations = 200
    e_maximum_steps = 100
    e_convergence_criterion = 0

    binary_latent_factor_model = q3.e_and_f(
        x=x,
        k=k,
        em_iterations=em_iterations,
        e_maximum_steps=e_maximum_steps,
        e_convergence_criterion=e_convergence_criterion,
        save_path=os.path.join(Q3_OUTPUT_FOLDER, "f"),
    )
    _ = q3.e_and_f(
        x=x,
        k=int(k * 1.5),
        em_iterations=em_iterations,
        e_maximum_steps=e_maximum_steps,
        e_convergence_criterion=e_convergence_criterion,
        save_path=os.path.join(Q3_OUTPUT_FOLDER, "f-larger-k"),
    )
    q3.g(
        x=x[:1, :],
        binary_latent_factor_model=binary_latent_factor_model,
        sigmas=[1, 2, 3],
        k=k,
        em_iterations=em_iterations,
        e_maximum_steps=e_maximum_steps,
        e_convergence_criterion=e_convergence_criterion,
        save_path=os.path.join(Q3_OUTPUT_FOLDER, "g"),
    )

    # Question 4
    Q4_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q4")
    if not os.path.exists(Q4_OUTPUT_FOLDER):
        os.makedirs(Q4_OUTPUT_FOLDER)
    ks = np.arange(int(k / 2), int(2 * k) + 1)
    q4.b(
        x=x,
        a_parameter=1,
        b_parameter=0,
        ks=ks,
        em_iterations=em_iterations,
        e_maximum_steps=e_maximum_steps,
        e_convergence_criterion=e_convergence_criterion,
        save_path=os.path.join(Q4_OUTPUT_FOLDER, "b"),
    )

    # Question 6
    Q6_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q6")
    if not os.path.exists(Q6_OUTPUT_FOLDER):
        os.makedirs(Q6_OUTPUT_FOLDER)
    q6.run(x, k, em_iterations, save_path=os.path.join(Q6_OUTPUT_FOLDER, "all"))
