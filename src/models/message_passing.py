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

    def set_xi(self, i, j, value):
        eta_values = 1 / (1 + np.exp(-value))
        eta_values[eta_values == 0] = 1e-10
        eta_values[eta_values == 1] = 1 - 1e-10
        self.eta_matrix[:, i, j] = eta_values

    def variational_expectation_step(
        self, x, binary_latent_factor_model: BoltzmannMachine
    ) -> List[float]:
        free_energy = [self.compute_free_energy(x, binary_latent_factor_model)]
        for i in range(self.k):
            for j in range(self.k):
                self.update_message(
                    binary_latent_factor_model,
                    x,
                    i,
                    j,
                )
            free_energy.append(self.compute_free_energy(x, binary_latent_factor_model))
        return free_energy

    def update_message(
        self,
        boltzmann_machine: BoltzmannMachine,
        x,
        start_node: int,
        end_node: int,
    ):
        if start_node != end_node:
            return self._update_binary_message(
                x, boltzmann_machine, start_node, end_node
            )
        else:
            return self._update_singleton_message(x, boltzmann_machine, start_node)

    def _update_binary_message(
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
        eta_j_not_i = boltzmann_machine.b_index(
            x=x, node_index=j
        ) + self.aggregate_incoming_binary_factor_messages(
            node_index=j, excluded_node_index=i
        )
        w_i_j = boltzmann_machine.w_matrix_index(i, j)
        self.set_xi(
            i=i,
            j=j,
            value=np.log(1 + np.exp(w_i_j + eta_i_not_j))
            - np.log(1 + np.exp(eta_i_not_j)),
        )
        self.set_xi(
            i=j,
            j=i,
            value=np.log(1 + np.exp(w_i_j + eta_j_not_i))
            - np.log(1 + np.exp(eta_j_not_i)),
        )

    def _update_singleton_message(
        self,
        x,
        boltzmann_machine: BoltzmannMachine,
        i: int,
    ):
        b_i = boltzmann_machine.b_index(x=x, node_index=i)
        self.set_xi(i=i, j=i, value=b_i)


def init_message_passing(k, n) -> MessagePassing:
    eta_matrix = np.random.random(size=(n, k, k))
    return MessagePassing(eta_matrix)
