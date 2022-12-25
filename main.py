import os
import pandas as pd
import numpy as np
from src.constants import CO2_FILE_PATH, IMAGES_FILE_PATH, OUTPUTS_FOLDER
from src.solutions import q2, q3, q4, q5, q6
from src.solutions.q2 import LinearRegressionParameters
from src.generate_images import generate_images


if __name__ == "__main__":
    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)

    with open(CO2_FILE_PATH) as file:
        lines = [line.rstrip().split() for line in file]

    # Question 2
    Q2_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q2")
    if not os.path.exists(Q2_OUTPUT_FOLDER):
        os.makedirs(Q2_OUTPUT_FOLDER)
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

    # # Question 3
    # Q3_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q3")
    # if not os.path.exists(Q3_OUTPUT_FOLDER):
    #     os.makedirs(Q3_OUTPUT_FOLDER)
    #
    # q3.learn_binary_factors(
    #     x=generate_images(),
    #     k=8,
    #     em_maximum_iterations=100,
    #     e_maximum_steps=100,
    #     e_convergence_criterion=0,
    # )
