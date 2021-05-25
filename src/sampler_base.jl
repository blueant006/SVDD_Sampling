using Clustering
using Distances
using JuMP
using MLKernels
using NearestNeighbors
using SVDD

using LinearAlgebra
using Statistics
using Random

abstract type Sampler end

DataSet = Array{Float64, 2}

function sample(sampler::Sampler, data::DataSet, labels::Vector{Symbol})::BitArray{1} end

struct SamplingException <: Exception
    msg
end

Base.showerror(io::IO, e::SamplingException) = print(io, "Exception during sampling: $(e.msg)")

include("density_util.jl")
include("pwc.jl")
include("sampler_util.jl")
include("sampler/RandomSampler.jl")
include("sampler/RAPID.jl")
include("sampler/FBPE.jl")

