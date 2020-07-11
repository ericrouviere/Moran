
# load code
using Distributed
using LinearAlgebra
using StatsBase
using Distributions
using JLD

numProcesses = 14
addprocs(numProcesses) # match numProcesses to --ntasks-per-node in .sbatch file
@everywhere using Distributed
@everywhere push!(LOAD_PATH, pwd()) # change pwd() to the path (a string) of Moran.jl
                                    # if it is not in the current working director
@everywhere using Moran


# parameters
P = 1000 # population size
w = 3 # alphabet size
L = 5 # protein length
μ = 0.001 # mutation rate
K = 2 # parameter from NK model
β = 10.0 # parameter for fitness sigmoid
σ = 0.0 # parameter for fitness sigmoid
numEnvs = 2 # number of environments
numGens = 1024 # number of generations to simulate.

# some checks to make sure things are proper.
@assert K+1 <= L # K cant be longer than sequence length
@assert numEnvs == 2 # can only handle 2 environments currently

numReps = 64
epochTimes = Float64.([0.25,0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]);

entropyData = zeros(numReps, length(epochTimes))
for (i,τ) in enumerate(epochTimes)
    println("working on $i")
    entropyData[:,i] = pmap(x -> wrapper(L,P,w,K,μ,β,σ,numEnvs,numGens,τ),1:numReps)
end

# save entropyData using the JLD package
save("entropyData.jld","1", entropyData)



