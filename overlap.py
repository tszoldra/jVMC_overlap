from jVMC.vqs import NQS
from jVMC.sampler import ExactSampler, MCSampler
from jVMC.mpi_wrapper import global_mean, global_sum
import jax.numpy as jnp
from custom_exact_sampler import ExactSamplerChunk
from typing import Union, List

# TODO implement it as an efficient jVMC.operator.Operator


def _overlap_autoregressive(psi_1: NQS, psi_2: NQS, sampler_1: MCSampler):
    assert psi_1.is_generator and psi_2.is_generator

    configs_1, logPsi_1, _ = sampler_1.sample()

    logPsi_2 = psi_2(configs_1)

    Z = jnp.exp(logPsi_2 - logPsi_1)
    Y = global_sum(Z) / Z.shape[-1]

    return Y


def _overlap_general(psi_1: NQS, psi_2: NQS, sampler_1: MCSampler,
                     sampler_2: MCSampler):

    configs_1, logPsi_1_configs_1, _ = sampler_1.sample()
    configs_2, logPsi_2_configs_2, _ = sampler_2.sample()

    logPsi_2_configs_1 = psi_2(configs_1)
    logPsi_1_configs_2 = psi_1(configs_2)

    Z = jnp.exp(logPsi_2_configs_1 - logPsi_1_configs_1)
    Y_1 = global_sum(Z) / Z.shape[-1]

    W = jnp.exp(logPsi_1_configs_2 - logPsi_2_configs_2)
    Y_2 = global_sum(W) / W.shape[-1]

    # phase = jnp.exp(0.5 * (jnp.log(Y_1 / jnp.abs(Y_1)) - jnp.log(Y_2 / jnp.abs(Y_2))))
    phase = Y_1 / jnp.abs(Y_1)
    Y = jnp.sqrt(jnp.abs(Y_1 * Y_2)) * phase

    return Y


def _overlap_exact(psi_2: NQS, sampler_1: ExactSampler):

    configs, logPsi_1, _ = sampler_1.sample()
    logPsi_2 = psi_2(configs)

    Y = global_sum(jnp.exp(jnp.conjugate(logPsi_1) + logPsi_2))

    return Y


def _overlap_exact_chunked(psi_2: NQS, sampler_1: ExactSamplerChunk, normalize=False, verbose=False):
    Y = jnp.zeros((1,), dtype=jnp.complex128)
    norm_psi1 = jnp.zeros((1,), dtype=jnp.float64)
    norm_psi2 = jnp.zeros((1,), dtype=jnp.float64)

    for chunkID in range(sampler_1.numChunks):
        if verbose:
            print(f"{chunkID=}")
        configs, logPsi_1, _ = sampler_1.sample(chunkID=chunkID)
        logPsi_2 = psi_2(configs)

        Y += global_sum(jnp.exp(jnp.conjugate(logPsi_1) + logPsi_2))

        if normalize:
            norm_psi1 += global_sum(jnp.abs(jnp.exp(logPsi_1))**2)
            norm_psi2 += global_sum(jnp.abs(jnp.exp(logPsi_2))**2)

    if normalize:
        return Y / jnp.sqrt(norm_psi1 * norm_psi2)
    else:
        return Y



def overlap(psi_1: NQS, psi_2: NQS, sampler_1: Union[ExactSampler, MCSampler, ExactSamplerChunk],
            sampler_2: Union[ExactSampler, MCSampler, ExactSamplerChunk] = None,
            force_general_algorithm=False, **kwargs):
    """
    Computes the complex overlap < psi_1 | psi_2 > between two neural quantum states.
    Samples configurations only from psi_1 if both NQS are generators/autoregressive/return normalized
    probability amplitudes.
    If ExactSampler object is used for sampler_1, the exact overlap (sum over the whole basis)
    is computed.
    If ExactSamplerChunk object is used for sampler1, the exact overlap is computed in batches/chunks to save memory.


    :param psi_1: First neural quantum state.
    :param psi_2: Second neural quantum state.
    :param sampler_1: Sampler associated with psi_1.
    :param sampler_2: Sampler associated with psi_2. Used only if one of NQS is not generator, or
    `force_general_algorithm=True`.
    :param force_general_algorithm: Whether to force the use of the general algorithm for unnormalized probability
    amplitudes.
    :param **kwargs: Extra parameters.

    :return: Overlap < psi_1 | psi_2 >.
    """
    if isinstance(sampler_1, ExactSampler):
        return _overlap_exact(psi_2, sampler_1, **kwargs)

    if isinstance(sampler_1, ExactSamplerChunk):
        if psi_1.is_generator and psi_2.is_generator:
            return _overlap_exact_chunked(psi_2, sampler_1, **kwargs)
        else:
            return _overlap_exact_chunked(psi_2, sampler_1, normalize=True, **kwargs)

    if psi_1.is_generator and psi_2.is_generator and not force_general_algorithm:
        return _overlap_autoregressive(psi_1, psi_2, sampler_1, **kwargs)

    return _overlap_general(psi_1, psi_2, sampler_1, sampler_2, **kwargs)
