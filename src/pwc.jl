abstract type ThresholdStrategy end

#=
Used threshold strategies in this code.

FixedThresholdStrategy(x): manually pass a float x

MaxDensityThresholdStrategy(eps) calculates the threshold by multiplying the maximum density with eps (eps must be in [0, 1]).

OutlierPercentageThresholdStrategy(eps) calculates the threshold as the eps-th quantile of the density (eps must be in [0, 1]). When one passes eps=nothing the threshold calculation takes the ground truth outlier percentage.

GroundTruthThresholdStrategy() calculates the threshold according to the ground truth.
=#


#This block of contains the calculation of the variables for kernel density estimation of the data and the outliers
# Initialisation of structure, it contains the variable "threshold", where this structure can be called for allocating the memory.
struct FixedThresholdStrategy <: ThresholdStrategy
    threshold::Real
end

#This function returns the threshold.
function calculate_threshold(threshold_strat::FixedThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    return threshold_strat.threshold
end

#This function checks the value of epsilon and throws the error if the value is not in between [0,1]
struct NumLabelTresholdStrategy <: ThresholdStrategy
    eps::Real
    function NumLabelTresholdStrategy(eps::Real)
        eps >= 0 || throw(ArgumentError("Negative eps: $(eps)."))
        eps <= 1 || throw(ArgumentError("Eps bigger than 1: $(eps)."))
        new(eps)
    end
end

#This function calculates threshold by multiplying the eps with the length of the labels.
function calculate_threshold(threshold_strat::NumLabelTresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    return threshold_strat.eps * length(labels)
end


#it checks the value of epsilon and throws the error if the value is not in between [0,1]
struct MaxDensityThresholdStrategy <: ThresholdStrategy
    eps::Real
    function MaxDensityThresholdStrategy(eps::Real)
        eps >= 0 || throw(ArgumentError("Negative eps: $(eps)."))
        eps <= 1 || throw(ArgumentError("Eps bigger than 1: $(eps)."))
        new(eps)
    end
end

# calculates the threshold by multiplying the maximum density with eps (eps must be in [0, 1]).
function calculate_threshold(threshold_strat::MaxDensityThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    return threshold_strat.eps * maximum(kde(kde_cache))
end

# calculates the threshold as the eps-th quantile of the density (eps must be in [0, 1]). When one passes eps=nothing the threshold calculation takes the ground truth outlier percentage.
struct OutlierPercentageThresholdStrategy <: ThresholdStrategy
    eps::Union{Real, Nothing}
    function OutlierPercentageThresholdStrategy(eps::Union{Real, Nothing}=nothing)
        if eps !== nothing
            eps >= 0 || throw(ArgumentError("Negative eps: $(eps)."))
            eps <= 1 || throw(ArgumentError("Eps bigger than 1: $(eps)."))
        end
        new(eps)
    end
end

function calculate_threshold(threshold_strat::OutlierPercentageThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    if threshold_strat.eps === nothing
        eps = count(x -> x .== :outlier, labels) / length(labels)
    else
        eps = threshold_strat.eps
    end
    d_x = sort(kde(kde_cache))
    n = length(labels)
    idx = max(1, min(n, floor(Int, eps * n)))
    return (d_x[idx] + d_x[min(n, idx+1)]) / 2
end

# calculates the threshold according to the ground truth.
struct GroundTruthThresholdStrategy <: ThresholdStrategy end


#KDE cache contains all the densities calculated in the data.
function calculate_threshold(threshold_strat::GroundTruthThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    eps = count(x -> x .== :outlier, labels) / length(labels)
    d_x = sort(kde(kde_cache))
    n = length(labels)
    idx = max(1, min(n, floor(Int, eps * n)))
    return (d_x[idx] + d_x[min(n, idx+1)]) / 2
end
function calculate_threshold(threshold_strat::ThresholdStrategy, data::DataSet, labels::Vector{Symbol},
                             gamma::Union{Real, Symbol})
    return calculate_threshold(threshold_strat, KDECache(data, gamma), labels)
end

#this function helps in differentiating between the inlier mask and outlier masks in the data.
function split_masks(threshold_strat::GroundTruthThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    threshold = calculate_threshold(threshold_strat, kde_cache, labels)
    inlier_mask = labels .== :inlier
    outlier_mask = .!inlier_mask
    return threshold, inlier_mask, outlier_mask
end

function split_masks(threshold_strat::Nothing, kde_cache::KDECache, labels::Vector{Symbol})
    return 0.0, trues(length(labels)), falses(length(labels))
end

function split_masks(threshold_strat::ThresholdStrategy, kde_cache::KDECache, labels::Vector{Symbol})
    threshold = calculate_threshold(threshold_strat, kde_cache, labels)
    d_x = kde(kde_cache)
    inlier_mask = d_x .>= threshold
    outlier_mask = .!inlier_mask
    return threshold, inlier_mask, outlier_mask
end

function split_masks(threshold_strat::ThresholdStrategy, data::DataSet, labels::Vector{Symbol}, gamma::Union{Nothing, Real, Symbol}=nothing)
    return split_data(threshold_strat, KDECache(data, gamma), labels)
end

#this structure contains the mutable composite objects
mutable struct PWC
    kde_cache::KDECache
    threshold_strat::ThresholdStrategy
    threshold::Union{Real, Nothing}
    function PWC(kde_cache::KDECache, threshold_strat::ThresholdStrategy=OutlierPercentageThresholdStrategy(),
                 threshold::Union{Real, Nothing}=nothing)
        return new(kde_cache, threshold_strat, threshold)
    end
end

function PWC(data, gamma::Union{Real, Symbol}, threshold_strat::ThresholdStrategy=OutlierPercentageThresholdStrategy())
    return PWC(KDECache(data, gamma), threshold_strat)
end

function calculate_threshold!(pwc::PWC, labels::Vector{Symbol})
    #storing calculated threshold in the object pwc
    pwc.threshold = calculate_threshold(pwc.threshold_strat, pwc.kde_cache, labels)
    return nothing
end

# This function returns the predicted values which has more value than the given threshold value.
function predict(pwc::PWC, data::DataSet)
    if pwc.threshold === nothing
        throw(ArgumentError("PWC threshold not initialized."))
    end
    d_x = kde(pwc.kde_cache, data) #kernal density estimation
    #filling the pred with predicted values.
    pred = fill(:outlier, size(data, 2))
    pred[d_x .> pwc.threshold] .= :inlier 
    return pred 
end
