#!/usr/bin/env python
# coding: utf-8

import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp
import flax.linen as nn

import numpy as np
import matplotlib.pyplot as plt

import jVMC

import sys
sys.path.insert(1, '../')

from overlap import overlap
from custom_exact_sampler import ExactSamplerChunk

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='Computes time evolution of random NQS with 1D TFIM Hamiltonian.\
                                       Calculates `numFidelityBins` fidelity and overlap estimates, each within a single bin,\
                                       where bins range from `fidelityBinsMin` to `fidelityBinsMax`. \
                                       Uses autoregressive Ansatz with multi-layer RNN.')
parser.add_argument('L', type=int, help='System size.')
parser.add_argument('g', type=float, help='Amplitude of sigma_x in the Hamiltonian.')
parser.add_argument('seed', type=int, help='Random seed.')
parser.add_argument('--n_steps', type=int, default=2000, help='Number of time steps.')
parser.add_argument('--n_Samples', type=int, default=200, help='Number of MC samples in the time evolution by SR.')
parser.add_argument('--hiddenSize', type=int, default=2, help='Hidden size of RNN.')
parser.add_argument('--depth', type=int, default=2, help='Vertical depth of RNN.')
parser.add_argument('--timeStep', type=float, default=5e-4, help='Time step of Euler integrator.')
parser.add_argument('--batchSize', type=int, default=8192, help='Batch size for time integrator. Use power of 2.')
parser.add_argument('--no_overlap_exact', action='store_true', help='Do not calculate the exact overlap \
                                                                    and always return crude MC estimate.')
parser.add_argument('--numChunks', type=int, default=1, help='Number of chunks of the full exact basis. Use even number.')
parser.add_argument('--numSamples_MC_overlap', type=int, default=512, help='Number of samples in the crude MC overlap estimation.')
parser.add_argument('--verbose', action='store_true', help='Print additional output.')
parser.add_argument('--numFidelityBins', type=int, default=10, help='Number of fidelity bins.')
parser.add_argument('--fidelityBinsMin', type=float, default=0.0, help='Fidelity bins minimal value.')
parser.add_argument('--fidelityBinsMax', type=float, default=1.0, help='Fidelity bins maximal value.')


args = parser.parse_args()



L = args.L
g = args.g
seed = args.seed

model_file = f'saved_models/ARNN_{L=}_{g=}_{seed=}.h5'

net_kwargs = dict(L=L, hiddenSize=args.hiddenSize, depth=args.depth, inputDim=2, actFun=nn.elu, logProbFactor=0.5,
                  cell='RNN', realValuedOutput=False)


n_steps = args.n_steps
n_Samples = args.n_Samples


net1 = jVMC.nets.RNN1DGeneral(**net_kwargs)
psi1 = jVMC.vqs.NQS(net1, seed=seed, batchSize=args.batchSize)  # Variational wave function

net2 = jVMC.nets.RNN1DGeneral(**net_kwargs)
psi2 = jVMC.vqs.NQS(net2, seed=seed, batchSize=args.batchSize)  # Variational wave function

s1 = psi1(jnp.ones((jVMC.global_defs.device_count(), 1, L), dtype=jnp.int32))
s2 = psi2(jnp.ones((jVMC.global_defs.device_count(), 1, L), dtype=jnp.int32))

print(f"The variational ansatz has {psi1.numParameters} parameters.")


# Set up sampler
sampler1 = jVMC.sampler.MCSampler(psi1, (L,), random.PRNGKey(seed),
                                  numSamples=n_Samples)
sampler2 = jVMC.sampler.MCSampler(psi2, (L,), random.PRNGKey(seed),
                                  numSamples=n_Samples)
#sampler2 = jVMC.sampler.ExactSampler(psi2, (L,), lDim=net_kwargs['inputDim'])

# overlap samplers
numSamples_overlap = args.numSamples_MC_overlap
sampler1_overlap = jVMC.sampler.MCSampler(psi1, (L,), random.PRNGKey(seed),
                                  numSamples=numSamples_overlap)
sampler2_overlap = jVMC.sampler.MCSampler(psi2, (L,), random.PRNGKey(seed),
                                  numSamples=numSamples_overlap)

if not args.no_overlap_exact:
    # sampler1_exact = jVMC.sampler.ExactSampler(psi1, (L,), lDim=net_kwargs['inputDim'])
    sampler1_exact = ExactSamplerChunk(psi1, (L,), lDim=net_kwargs['inputDim'], numChunks=args.numChunks)

# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()
for l in range(L):
    hamiltonian.add(jVMC.operator.scal_opstr(-1., (jVMC.operator.Sz(l), jVMC.operator.Sz((l + 1) % L))))
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l), )))



# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(sampler2, rhsPrefactor=1.j,
                                   svdTol=1e-6, diagonalShift=0., makeReal='imag', )

stepper = jVMC.util.stepper.Euler(timeStep=args.timeStep)  # ODE integrator



outp = jVMC.util.OutputManager(dataFileName=model_file, append=True)
outp.write_network_checkpoint(-1, psi1.get_parameters())  # psi1 is saved as time = -1.


fidelity_bins = np.linspace(args.fidelityBinsMin, args.fidelityBinsMax, args.numFidelityBins)
Fs_found = [False for _ in range(len(fidelity_bins))]

ov_exact_networks = []

def get_fidelity_idx(f):
    return int( (f - args.fidelityBinsMin) / (args.fidelityBinsMax - args.fidelityBinsMin) * len(fidelity_bins) )


i=0
for n in range(n_steps):
    dp, _ = stepper.step(0, tdvpEquation, psi2.get_parameters(),
                         hamiltonian=hamiltonian, psi=psi2, numSamples=None)
    
    fidelity_MC = np.abs(overlap(psi1, psi2, sampler1_overlap))**2
    
    idx_fidelity_MC = get_fidelity_idx(fidelity_MC)
    
    if args.verbose:
        print(f'timestep {n=}')
    
    if idx_fidelity_MC >= len(fidelity_bins) or idx_fidelity_MC<0:
        psi2.set_parameters(dp)
        continue # we do not want 1.0 exactly at the beginning of the time evolution
    
    if not Fs_found[idx_fidelity_MC]:
        ov_exact = overlap(psi1, psi2, sampler1_exact, verbose=args.verbose)
        fidelity_exact = np.abs(ov_exact)**2
        
        idx_fidelity_exact = get_fidelity_idx(fidelity_exact[0])
            
        if not Fs_found[idx_fidelity_exact]:
            outp.write_network_checkpoint(i, psi2.get_parameters())
            ov_exact_networks.append(ov_exact)
            Fs_found[idx_fidelity_exact] = True
            i += 1
            
            if args.verbose:
                print(f'{n=} ================== saving {fidelity_exact=} ========== remaining {len(fidelity_bins) - np.sum(Fs_found)} fidelity bins')

    psi2.set_parameters(dp)
    
    if np.all(np.array(Fs_found)):
        break
    


outp.write_dataset('overlap_exact', np.array(ov_exact_networks))
