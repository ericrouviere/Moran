module Moran

## load packages #############
using LinearAlgebra
using StatsBase
using Distributions

# make function accessible by exporting them
export makeRandNKtable, genRandSeq, computeFitness,
    computeEntropy, computeFreqs, pickWinner, pickLoser,
    sigmoid, mutate, randp, evolve, buildDict, finduniques,
    wrapper

########################################
#### evolution functions  ###########
########################################


function makeRandNKtable(K::Int, w::Int)
    # make multi dim array of random normals
    # where array has K+1 dims and w elements along each dim
    return randn(Tuple(fill(w, K+1)))
end


function genRandSeq(seqLen::Int, alphabetSize::Int)
    # generate random sequence of integers 1 through alphabetSize.
    return rand(1:alphabetSize, seqLen)
end

function computeFitness(seq::AbstractArray, env::AbstractArray,
                        K::Int, β::Number, σ::Number)
    # compute the fitness of a sequence in a given environment

    # First compute the phenotype from the NK model
    phenotype = 0.0
    extendedSeq = [seq; seq[1:K]]
    for i in eachindex(seq)
        indices = extendedSeq[i:i+K]
        phenotype += env[indices...]
    end

    # the phenotype to fitness map is sigmoidal
    fitness = sigmoid(phenotype, β, σ)
    return fitness
end


function sigmoid(x::Number, β::Number, σ::Number)
    return 1 / (1 + exp(- β * (x-σ) ))
end

function updateFitnesses!(fits::Vector{Float64},
                          seqs::Matrix{Int},
                          env::Array,
                          K::Int, β::Number, σ::Number)
    # computes and updates the fitnesses for all sequences based
    # on environment. Inplace function

    for i in eachindex(fits)
        seq = @view seqs[:,i]
        fits[i] = computeFitness(seq, env, K, β, σ)
    end
    return nothing
end


function pickWinner(fits::Vector{Float64})::Int
    # choose who get to reproduce based on the 
    # fitnesses.
    
    # this function is optimizied to be fast

    c1=0.0
    c2=0.0
    # compute the sum of the fits vector
    @inbounds @simd for i in eachindex(fits)
        @fastmath c1 += fits[i] 
    end

    # pick random index using fits vector as distribution
    r = c1*rand()
    @inbounds for (i,p) in enumerate(fits)
        c2 += p
        if r <= c2
            return i
        end
    end
    return length(fits)
end

function pickLoser(numSeqs::Int)::Int
    # pick loser randomly 
    return rand(1:numSeqs)
end


function mutate!(seq::AbstractArray, mutationRate::Number, numTypes::Int)
    # mutates sequence seq inplace.
    seqLen = length(seq)
    numMuts = rand(Binomial(seqLen,mutationRate))
    positions = sample(1:seqLen, numMuts, replace=false)
    for i in positions
        seq[i] = rand(setdiff(1:numTypes, seq[i]))
    end
    return nothing
end

function switchEnv(envIndex::Int)
    # switch the environment index from,
    # 1 -> 2  or  2 -> 1
    return 3 - envIndex
end

function evolve(seqs::Matrix{Int}, # sequences
                fits::Vector{Float64}, # fitnesses
                envs::Array{<:Array{Float64}}; # environments
                μ::Number=0.001, #mutation rate
                β::Number=1.0, # parameter for sigmoid
                σ::Number=0.0, # parameter for sigmoid
                numGens::Int=1000, # number of generations
                epochTime::Number=100000, # epoch time
                envIndex::Int=1) # initial enviroment

    # evolve the population using moran dynamics.

    # get parameters
    w = size(envs[1],1) # alphabet size
    K = ndims(envs[1]) - 1 # K parameter of NK model
    P = length(fits) # population size
    L = size(seqs,1) # length of sequence

    # intialize arrays to save data along the way
    seqsData = zeros(Int, L, P, numGens+1)
    fitsData = zeros(P, numGens+1)
    saveData!(seqsData, fitsData, seqs, fits, 1)

    for i in 1:numGens
        for j in 1:P
            # switch environment periodically
            if ((P*(i-1)+j) % Int(round(epochTime*P)) ) == 0 
                envIndex = switchEnv(envIndex)
                # update fitnesses for new environment
                updateFitnesses!(fits, seqs, envs[envIndex], K, β, σ)
            end
            
            birthIndex = pickWinner(fits)
            deathIndex = pickLoser(P)
            seqs[:,deathIndex] .= seqs[:,birthIndex]
            seq = @view seqs[:,deathIndex]
            mutate!(seq, μ, w)
            fits[deathIndex] = computeFitness(seq, envs[envIndex], K, β, σ)
        end
        saveData!(seqsData, fitsData, seqs, fits, i+1)
    end
    return seqsData, fitsData
end


function saveData!(seqsData::Array, fitsData::Array,
                   seqs::Matrix{Int}, fits::Vector{Float64}, i::Int)
    # save the data from seqs and fits to
    # seqsData and fitsData. Inplace function.
    seqsData[:,:,i] .= seqs
    fitsData[:,i] .= fits
    return nothing
end


##########################################
##### Analysis functions #################
###########################################


function finduniques(seqsData::Array{Int,3})
    # find the unique sequences in the sequences data structure
    # return an 1-D array of uniques sequences
    #
    # CURRENTLY VERY SLOW 

    uniqs = Array{Int,1}[]
    L,P,numGens = size(seqsData)
    @inbounds for i in 1:numGens, j in 1:P
        seq = seqsData[:,j,i]
        if !(seq in uniqs)
            push!(uniqs, seq)
        end
    end
    return uniqs
end

function computeFreqs(seqsData::Array{Int, 3}, uniqs::Array{Array{Int,1}})
    # compute the number of each uniques sequence in the seqsData array

    # return an array of integers, each column gives the number of a
    # particular sequence as function generatation number.

    # CURRENTLY VERY SLOW

    numGens = size(seqsData,3)
    numUniqs = length(uniqs)
    P = size(seqsData,2)

    freqs = zeros(Int, numGens, numUniqs)
    D = buildDict(uniqs);
    for i in 1:numGens
        seqs = seqsData[:,:,i]
        for j in 1:P
            seq = seqs[:,j]
            freqs[i,D[seq]] += 1
        end
    end
    return freqs
end

function buildDict(uniqs::Array{Array{Int,1}})
    # make Dictionary that relates a sequence to an index in 
    # the uniqs sequence array.
    D = Dict()
    for (i,s) in enumerate(uniqs)
        D[s] = i
    end
    return D
end

function computeEntropy(p::Vector{Float64})
    # computes the entropy of the distrubtion p
    p .= p ./ sum(p)
    S = 0.0
    @inbounds for i in eachindex(p)
        P = p[i]
        S -= ifelse(P>0,P*log(P), 0.0)
    end
    return S
end


function wrapper(L::Int, P::Int, w::Int, K::Int, μ::Number,
                 β::Number, σ::Number, numEnvs::Int,
                 numGens::Int, epochTime::Number)

    # this function puts a wrapper of the work flow to compute entropy
    # Build environments, initialized arrays, evolve population,
    # compute frequencies and compute entropy

    envs = [makeRandNKtable(K,w) for i in 1:numEnvs]
    seqs0 = zeros(Int, L, P) # population
    fits0 = zeros(P) # fitnesses
    # start out with random population
    for i in 1:P
        seq = genRandSeq(L,w)
        seqs0[:,i] = seq
        fits0[i] = computeFitness(seq, envs[1], K, β, σ)
    end
    seqsData, fitsData = evolve(seqs0, fits0, envs, μ=μ, β=β, σ=σ, numGens=numGens, epochTime=epochTime);

    uniqs = finduniques(seqsData)
    freqs = computeFreqs(seqsData, uniqs);
    probs = Float64.(vec(sum(freqs[24:end,:], dims=1)))  # the 24 means that you dont 
                                                         # consider the first 24 generations.
    S = computeEntropy(probs)
    return S
end


end

