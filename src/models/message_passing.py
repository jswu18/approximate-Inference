from typing import List

import numpy as np

from src.models.binary_latent_factor_model_approximation import (
    BinaryLatentFactorApproximation,
)
from src.models.boltzmann_machine import BoltzmannMachine


class MessagePassing(BinaryLatentFactorApproximation):
    """
    eta_matrix:  of parameters eta_matrix[n, i, j]
                off diagonals corresponds to \tilda{g}_{ij, \neg s_i}(s_j) for data point n
                diagonals correspond to \tilda{f}_{i}(s_i)
                (number_of_points, number_of_latent_variables, number_of_latent_variables)
    """

    def __init__(self, eta_matrix: np.ndarray):
        self.eta_matrix = eta_matrix

    @property
    def lambda_matrix(self) -> np.ndarray:
        lambda_matrix = 1 / (1 + np.exp(-self.xi.sum(axis=1)))
        lambda_matrix[lambda_matrix == 0] = 1e-10
        lambda_matrix[lambda_matrix == 1] = 1 - 1e-10
        return lambda_matrix

    @property
    def xi(self) -> np.ndarray:
        return np.log(np.divide(self.eta_matrix, 1 - self.eta_matrix))

    def aggregate_incoming_binary_factor_messages(
        self, node_index: int, excluded_node_index: int
    ) -> np.ndarray:
        # (number_of_points, )
        #  exclude message from excluded_node_index -> node_index
        return (
            np.sum(self.xi[:, :excluded_node_index, node_index], axis=1)
            + np.sum(self.xi[:, excluded_node_index + 1 :, node_index], axis=1)
        ).reshape(
            -1,
        )

    @staticmethod
    def calculate_eta(xi: np.ndarray) -> np.ndarray:
        eta = 1 / (1 + np.exp(-xi))
        eta[eta == 0] = 1e-10
        eta[eta == 1] = 1 - 1e-10
        return eta

    def variational_expectation_step(
        self, x: np.ndarray, binary_latent_factor_model: BoltzmannMachine
    ) -> List[float]:
        free_energy = [self.compute_free_energy(x, binary_latent_factor_model)]
        for i in range(self.k):
            xi_new_ii = self.calculate_singleton_message_update(
                boltzmann_machine=binary_latent_factor_model,
                x=x,
                i=i,
            )
            self.eta_matrix[:, i, i] = self.calculate_eta(xi_new_ii)
            free_energy.append(self.compute_free_energy(x, binary_latent_factor_model))

            for j in range(i):
                xi_new_ij = self.calculate_binary_message_update(
                    boltzmann_machine=binary_latent_factor_model,
                    x=x,
                    i=i,
                    j=j,
                )
                self.eta_matrix[:, i, j] = self.calculate_eta(xi_new_ij)
                xi_new_ji = self.calculate_binary_message_update(
                    boltzmann_machine=binary_latent_factor_model,
                    x=x,
                    i=j,
                    j=i,
                )
                self.eta_matrix[:, j, i] = self.calculate_eta(xi_new_ji)
                free_energy.append(
                    self.compute_free_energy(x, binary_latent_factor_model)
                )
        return free_energy

    def calculate_binary_message_update(
        self,
        x: np.ndarray,
        boltzmann_machine: BoltzmannMachine,
        i: int,
        j: int,
    ) -> float:
        eta_i_not_j = boltzmann_machine.b_index(
            x=x, node_index=i
        ) + self.aggregate_incoming_binary_factor_messages(
            node_index=i, excluded_node_index=j
        )
        w_i_j = boltzmann_machine.w_matrix_index(i, j)
        return np.log(1 + np.exp(w_i_j + eta_i_not_j)) - np.log(1 + np.exp(eta_i_not_j))

    @staticmethod
    def calculate_singleton_message_update(
        x: np.ndarray,
        boltzmann_machine: BoltzmannMachine,
        i: int,
    ) -> float:
        return boltzmann_machine.b_index(x=x, node_index=i)


def init_message_passing(k, n) -> MessagePassing:
    eta_matrix = np.random.random(size=(n, k, k))
    return MessagePassing(eta_matrix)
