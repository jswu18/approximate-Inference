from typing import List

import numpy as np

from src.models.binary_latent_factor_approximations.abstract_binary_latent_factor_approximation import (
    AbstractBinaryLatentFactorApproximation,
)
from src.models.binary_latent_factor_models.boltzmann_machine import BoltzmannMachine


class MessagePassingApproximation(AbstractBinaryLatentFactorApproximation):
    """
    bernoulli_parameter_matrix (theta): matrix of parameters bernoulli_parameter_matrix[n, i, j]
                off diagonals corresponds to \tilda{g}_{ij, \neg s_i}(s_j) for data point n
                diagonals correspond to \tilda{f}_{i}(s_i)
                (number_of_points, number_of_latent_variables, number_of_latent_variables)
    """

    def __init__(self, bernoulli_parameter_matrix: np.ndarray):
        self.bernoulli_parameter_matrix = bernoulli_parameter_matrix

    @property
    def lambda_matrix(self) -> np.ndarray:
        """
        Aggregate messages and compute parameter for Bernoulli distribution
        :return:
        """
        lambda_matrix = 1 / (1 + np.exp(-self.natural_parameter_matrix.sum(axis=1)))
        lambda_matrix[lambda_matrix == 0] = 1e-10
        lambda_matrix[lambda_matrix == 1] = 1 - 1e-10
        return lambda_matrix

    @property
    def natural_parameter_matrix(self) -> np.ndarray:
        """
        The matrix containing natural parameters (eta) of each factor
                off diagonals corresponds to \tilda{g}_{ij, \neg s_i}(s_j) for data point n
                diagonals correspond to \tilda{f}_{i}(s_i)
                (number_of_points, number_of_latent_variables, number_of_latent_variables)
        :return:
        """
        return np.log(
            np.divide(
                self.bernoulli_parameter_matrix, 1 - self.bernoulli_parameter_matrix
            )
        )

    def aggregate_incoming_binary_factor_messages(
        self, node_index: int, excluded_node_index: int
    ) -> np.ndarray:
        # (number_of_points, )
        #  exclude message from excluded_node_index -> node_index
        return (
            np.sum(
                self.natural_parameter_matrix[:, :excluded_node_index, node_index],
                axis=1,
            )
            + np.sum(
                self.natural_parameter_matrix[:, excluded_node_index + 1 :, node_index],
                axis=1,
            )
        ).reshape(
            -1,
        )

    @staticmethod
    def calculate_bernoulli_parameter(
        natural_parameter_matrix: np.ndarray,
    ) -> np.ndarray:
        bernoulli_parameter = 1 / (1 + np.exp(-natural_parameter_matrix))
        bernoulli_parameter[bernoulli_parameter == 0] = 1e-10
        bernoulli_parameter[bernoulli_parameter == 1] = 1 - 1e-10
        return bernoulli_parameter

    def variational_expectation_step(
        self, x: np.ndarray, binary_latent_factor_model: BoltzmannMachine
    ) -> List[float]:
        """
        Iteratively update singleton and binary factors
        :param x: data matrix (number_of_points, number_of_dimensions)
        :param binary_latent_factor_model: a binary_latent_factor_model
        :return: free energies after each update
        """
        free_energy = [self.compute_free_energy(x, binary_latent_factor_model)]
        for i in range(self.k):
            # singleton factor update
            natural_parameter_ii = self.calculate_singleton_message_update(
                boltzmann_machine=binary_latent_factor_model,
                x=x,
                i=i,
            )
            self.bernoulli_parameter_matrix[
                :, i, i
            ] = self.calculate_bernoulli_parameter(natural_parameter_ii)
            free_energy.append(self.compute_free_energy(x, binary_latent_factor_model))

            for j in range(i):
                # binary factor update
                natural_parameter_ij = self.calculate_binary_message_update(
                    boltzmann_machine=binary_latent_factor_model,
                    x=x,
                    i=i,
                    j=j,
                )
                self.bernoulli_parameter_matrix[
                    :, i, j
                ] = self.calculate_bernoulli_parameter(natural_parameter_ij)
                natural_parameter_ji = self.calculate_binary_message_update(
                    boltzmann_machine=binary_latent_factor_model,
                    x=x,
                    i=j,
                    j=i,
                )
                self.bernoulli_parameter_matrix[
                    :, j, i
                ] = self.calculate_bernoulli_parameter(natural_parameter_ji)
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
        """
        Calculate new parameters for a binary factored message.

        :param x: data matrix (number_of_points, number_of_dimensions)
        :param boltzmann_machine: Boltzmann machine model
        :param i: starting node for the message
        :param j: ending node for the message
        :return: new parameter from aggregating incoming messages
        """
        natural_parameter_i_not_j = boltzmann_machine.b_index(
            x=x, node_index=i
        ) + self.aggregate_incoming_binary_factor_messages(
            node_index=i, excluded_node_index=j
        )
        w_i_j = boltzmann_machine.w_matrix_index(i, j)
        return np.log(1 + np.exp(w_i_j + natural_parameter_i_not_j)) - np.log(
            1 + np.exp(natural_parameter_i_not_j)
        )

    @staticmethod
    def calculate_singleton_message_update(
        x: np.ndarray,
        boltzmann_machine: BoltzmannMachine,
        i: int,
    ) -> float:
        """
        Calculate the parameter update for the singleton message.
        Note that this does not require any approximation.

        :param x: data matrix (number_of_points, number_of_dimensions)
        :param boltzmann_machine: Boltzmann machine model
        :param i: node to update
        :return: new parameter
        """
        return boltzmann_machine.b_index(x=x, node_index=i)


def init_message_passing(k: int, n: int) -> MessagePassingApproximation:
    """
    Message passing initialisation

    :param k: number of latent variables
    :param n: number of data points
    :return: message passing
    """
    bernoulli_parameter_matrix = np.random.random(size=(n, k, k))
    return MessagePassingApproximation(bernoulli_parameter_matrix)
