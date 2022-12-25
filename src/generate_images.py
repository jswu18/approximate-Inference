import numpy as np

from src.constants import DEFAULT_SEED, M1, M2, M3, M4, M5, M6, M7, M8


def generate_images(n: int = 400, seed: int = DEFAULT_SEED):
    """

    :param n: number of data points
    :param seed: random seed
    :return:
    """
    d = 16  # dimensionality of the data
    np.random.seed(seed)

    # Define the basic shapes of the features
    number_of_features = 8  # number of features
    rr = (
        0.5 + np.random.rand(number_of_features, 1) * 0.5
    )  # weight of each feature between 0.5 and 1
    mut = np.array(
        [
            rr[0] * M1,
            rr[1] * M2,
            rr[2] * M3,
            rr[3] * M4,
            rr[4] * M5,
            rr[5] * M6,
            rr[6] * M7,
            rr[7] * M8,
        ]
    )
    s = (
        np.random.rand(n, number_of_features) < 0.3
    )  # each feature occurs with prob 0.3 independently

    # Generate Data - The Data is stored in Y

    return np.dot(s, mut) + np.random.randn(n, d) * 0.1  # some Gaussian noise is added
