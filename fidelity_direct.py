from jVMC.vqs import NQS
from jVMC.sampler import MCSampler
from jVMC.mpi_wrapper import global_mean
import jax.numpy as jnp

from typing import Union

# TODO implement it as an efficient jVMC.operator.Operator


def fidelity_direct(psi_1: NQS, psi_2: NQS, sampler_1: MCSampler,
                     sampler_2: MCSampler):
    """
    Computes the complex estimate of fidelity  < psi_1 | psi_2 > < psi_2 | psi_1 > between two neural quantum states.
    Samples configurations from both psi_1 and psi_2. States need not be normalized.

    :param psi_1: First neural quantum state.
    :param psi_2: Second neural quantum state.
    :param sampler_1: Sampler associated with psi_1.
    :param sampler_2: Sampler associated with psi_2`.
    :return: Fidelity (< psi_1 | psi_2 >) estimate (a complex number).
    """

    configs_1, logPsi_1_configs_1, _ = sampler_1.sample()
    configs_2, logPsi_2_configs_2, _ = sampler_2.sample()

    logPsi_2_configs_1 = psi_2(configs_1)
    logPsi_1_configs_2 = psi_1(configs_2)

    Z = jnp.exp(logPsi_2_configs_1 - logPsi_1_configs_1)
    Y_1 = global_mean(Z)

    W = jnp.exp(logPsi_1_configs_2 - logPsi_2_configs_2)
    Y_2 = global_mean(W)

    return Y_1 * Y_2