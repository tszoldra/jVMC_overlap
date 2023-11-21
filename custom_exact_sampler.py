from jVMC.vqs import NQS
from jVMC.mpi_wrapper import global_mean, global_sum
import jax.numpy as jnp

from typing import Union

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax import vmap, jit

import jVMC.mpi_wrapper as mpi

from functools import partial

import time

import jVMC.global_defs as global_defs


class ExactSamplerChunk:
    """Class for partial enumeration of basis states.

    This class generates a full basis of the many-body Hilbert space in chunks. Thereby, it \
    allows to exactly perform sums over the full Hilbert space instead of stochastic \
    sampling.

    Initialization arguments:
        * ``net``: Network defining the probability distribution.
        * ``sampleShape``: Shape of computational basis states.
        * ``lDim``: Local Hilbert space dimension.
        * ``numChunks``: Number of chunks. numChunks * batchSize gives the total number of elements in the basis (2^L for spins).
        * ``chunkID``: Current chunk ID. Can range from 0 to numChunks-1.
    """

    def __init__(self, net, sampleShape, lDim=2, logProbFactor=0.5, numChunks=1):

        self.psi = net
        self.N = jnp.prod(jnp.asarray(sampleShape))
        self.sampleShape = sampleShape
        self.lDim = lDim
        self.logProbFactor = logProbFactor
        self.numChunks = numChunks
        self.chunkSize = int(self.lDim ** self.N / self.numChunks)
        assert self.lDim ** self.N / self.numChunks == self.chunkSize
        assert self.numChunks % 2 == 0 or numChunks == 1

        # pmap'd member functions
        self._get_basis_ldim2_pmapd = global_defs.pmap_for_my_devices(self._get_basis_ldim2, in_axes=(0, 0, None),
                                                                      static_broadcasted_argnums=2)
        self._get_basis_pmapd = global_defs.pmap_for_my_devices(self._get_basis, in_axes=(0, 0, None, None),
                                                                static_broadcasted_argnums=(2, 3))
        #self.get_basis()
        # Make sure that net params are initialized
        self.psi(jnp.ones((global_defs.device_count(), 1, self.N), dtype=jnp.int32))

    def get_basis(self, chunkID):

        myNumStates = mpi.distribute_sampling(self.chunkSize)
        myFirstState = mpi.first_sample_id()

        deviceCount = global_defs.device_count()

        self.numStatesPerDevice = [(myNumStates + deviceCount - 1) // deviceCount] * deviceCount
        self.numStatesPerDevice[-1] += myNumStates - deviceCount * self.numStatesPerDevice[0]
        self.numStatesPerDevice = jnp.array(self.numStatesPerDevice)

        totalNumStates = deviceCount * self.numStatesPerDevice[0]

        intReps = jnp.arange(myFirstState + chunkID * self.chunkSize, myFirstState + chunkID * self.chunkSize + totalNumStates)
        intReps = intReps.reshape((global_defs.device_count(), -1))
        basis = jnp.zeros(intReps.shape + (self.N,), dtype=np.int32)
        if self.lDim == 2:
            basis = self._get_basis_ldim2_pmapd(basis, intReps, self.sampleShape)
        else:
            basis = self._get_basis_pmapd(basis, intReps, self.lDim, self.sampleShape)

        return basis

    def _get_basis_ldim2(self, states, intReps, sampleShape):

        def make_state(state, intRep):
            def for_fun(i, x):
                return (jax.lax.cond(x[1] >> i & 1, lambda x: x[0].at[x[1]].set(1), lambda x: x[0], (x[0], i)), x[1])

            (state, _) = jax.lax.fori_loop(0, state.shape[0], for_fun, (state, intRep))

            return state.reshape(sampleShape)

        basis = jax.vmap(make_state, in_axes=(0, 0))(states, intReps)

        return basis

    def _get_basis(self, states, intReps, lDim, sampleShape):

        def make_state(state, intRep):
            def scan_fun(c, x):
                locState = c % lDim
                c = (c - locState) // lDim
                return c, locState

            _, state = jax.lax.scan(scan_fun, intRep, state)

            return state[::-1].reshape(sampleShape)

        basis = jax.vmap(make_state, in_axes=(0, 0))(states, intReps)

        return basis


    def sample(self, parameters=None, numSamples=None, multipleOf=None, chunkID=None):
        """Return chunk computational basis states.

        Sampling is automatically distributed accross MPI processes and available \
        devices.

        Arguments:
            * ``parameters``: dict with ``chunkID``: Current chunk ID. Can range from 0 to numChunks-1.
            * ``numSamples``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.
            * ``multipleOf``: Dummy argument to provide identical interface as the \
            ``MCSampler`` class.

        Returns:
            ``configs, logPsi, None``: All computational basis configurations in the chunk, \
            corresponding wave function coefficients, None.
        """
        basis = self.get_basis(chunkID)
        logPsi = self.psi(basis)

        return basis, logPsi, None

    def set_number_of_samples(self, N):
        pass

# ** end class ExactSampler
