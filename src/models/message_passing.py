import numpy as np
from src.models.binary_latent_factor_model import BinaryLatentFactorApproximation
from src.models.boltzmann_machine import BoltzmannMachine

from typing import List


class MessagePassing(BinaryLatentFactorApproximation):
    """
    eta_matrix: off diagonal matrix of parameters eta_matrix[n, i, j] corresponds to \tilda{g}_{ij, \neg s_i}(s_j) for
                data point n
                (number_of_points, number_of_latent_variables, number_of_latent_variables)
    """

    def __init__(self, eta_matrix: np.ndarray):
        self.eta_matrix = eta_matrix

    @property
    def lambda_matrix(self):
        return 1 / (1 + np.exp(-self.xi.sum(axis=1)))

    @property
    def xi(self):
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
    def calculate_eta(xi):
        eta = 1 / (1 + np.exp(-xi))
        eta[eta == 0] = 1e-10
        eta[eta == 1] = 1 - 1e-10
        return eta

    def variational_expectation_step(
        self, x, binary_latent_factor_model: BoltzmannMachine
    ) -> List[float]:
        free_energy = [self.compute_free_energy(x, binary_latent_factor_model)]
        for i in range(self.k):
            for j in range(self.k):
                xi_new = self.calculate_message_update(
                    boltzmann_machine=binary_latent_factor_model,
                    x=x,
                    start_node=i,
                    end_node=j,
                )
                self.eta_matrix[:, i, j] = self.calculate_eta(xi_new)
            free_energy.append(self.compute_free_energy(x, binary_latent_factor_model))
        return free_energy

    def calculate_message_update(
        self,
        boltzmann_machine: BoltzmannMachine,
        x,
        start_node: int,
        end_node: int,
    ):
        if start_node != end_node:
            return self._calculate_binary_message_update(
                x, boltzmann_machine, start_node, end_node
            )
        else:
            return self._calculate_singleton_message_update(
                x, boltzmann_machine, start_node
            )

    def _calculate_binary_message_update(
        self,
        x,
        boltzmann_machine: BoltzmannMachine,
        i: int,
        j: int,
    ):
        eta_i_not_j = boltzmann_machine.b_index(
            x=x, node_index=i
        ) + self.aggregate_incoming_binary_factor_messages(
            node_index=i, excluded_node_index=j
        )
        w_i_j = boltzmann_machine.w_matrix_index(i, j)
        return np.log(1 + np.exp(w_i_j + eta_i_not_j)) - np.log(1 + np.exp(eta_i_not_j))

    def _calculate_singleton_message_update(
        self,
        x,
        boltzmann_machine: BoltzmannMachine,
        i: int,
    ):
        return boltzmann_machine.b_index(x=x, node_index=i)


def init_message_passing(k, n) -> MessagePassing:
    eta_matrix = np.random.random(size=(n, k, k))
    return MessagePassing(eta_matrix)