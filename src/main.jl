
include("sampler_base.jl")

import JuMP
import Ipopt
import MLKernels
import SVDD

using Random
using Test
Random.seed!(0)

TEST_SOLVER =  JuMP.with_optimizer(Ipopt.Optimizer, print_level=0)

@testset "Sampling test" begin
    Random.seed!(0)
    #randn is used to generate a normal distribution 
    #hcat is used for 2nd dimeension concatenation
    d, l = hcat(randn(5, 9500), randn(5, 500) .+ 2), fill(:inlier, 10000)
    
    l[end-499:end] .= :outlier
   # print(d)
    #print(l)
    gamma = 0.5
    c = KDECache(d, gamma)
    chunk_mask = trues(length(l))
    chunk_mask[1:5] .= false
    threshold_strat = MaxDensityThresholdStrategy(0.5)
    threshold_strat_gt = GroundTruthThresholdStrategy()

    init_strat = SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(MLKernels.GaussianKernel(gamma)),
                                             SVDD.FixedCStrategy(0.5))

    #test module is useful for unit testing
    @time begin
        @testset "RandomSampler" begin
            @test_throws ArgumentError RandomRatioSampler(-1)
            #sampling ratio is 0.5
            s = RandomRatioSampler(0.5)
            sample_mask = sample(s, d, l)
             print(sample_mask)
            s = RandomRatioSampler(0.5)
            sample_mask = sample(s, d, l)
            #print("In random sample % of samples is:")
            #println(count(sample_mask)/length(sample_mask))
            #@test length(sample_mask) == length(l)
            @test !all(sample_mask)
        end
    end
    @time begin
        @testset "FBPE" begin
            @test_throws ArgumentError FBPE(-1)
            @test_throws ArgumentError FBPE(0)
            s = FBPE(100)
            sample_mask = sample(s, d, l)
            #train_and_find_support_vectors(init_strat, d, TEST_SOLVER )
            #print(sample_mask)
            #print(length(sample_mask))
            #print("In FBPE sample % of samples is:")
            #println(count(sample_mask))
            #println(count(sample_mask)/length(sample_mask))
            @test length(sample_mask) == length(l)
            @test !all(sample_mask)
        end
    end
    @time begin

        @testset "RAPID" begin
            s = RAPID(threshold_strat, :test)
            @test_throws ArgumentError sample(s, d, l)
            s = RAPID(threshold_strat, c)
            sample_mask = sample(s, d, l)
            #print(sample_mask)
            @test length(sample_mask) == length(l)
            @test !all(sample_mask)
            s = RAPID(threshold_strat_gt, c)
            sample_mask = sample(s, d, l)
            #print("In RAPID sample % of samples is:")
            #println(count(sample_mask))
            #println(count(sample_mask)/length(sample_mask))
            #print(length(sample_mask))
            #print(sample_mask)
            @test length(sample_mask) == length(l)
            @test !all(sample_mask)
            _, inlier_mask, _ = split_masks(threshold_strat_gt, c, l)
            @test all(inlier_mask[sample_mask])
        end
    end
    
end
@testset "Density Testing" begin
    Random.seed!(0)
    d, l = hcat(randn(5, 950), randn(5, 50) .+ 2), fill(:inlier, 1000)
    l[end-49:end] .= :outlier
    gamma = 0.5
    c = KDECache(d, gamma)

    @testset "Threshold Strategies" begin
        @testset "FixedThresholdStrategy" begin
            n = 3
            s = FixedThresholdStrategy(n)
            @test n == calculate_threshold(s, d, l, gamma)
            @test n == calculate_threshold(s, c, l)
        end

        for x in [NumLabelTresholdStrategy, MaxDensityThresholdStrategy, OutlierPercentageThresholdStrategy]
            @testset "$x" begin
                @test_throws ArgumentError x(-1)
                @test_throws ArgumentError x(2)
                s = x(0.1)
                t, i, o = split_masks(s, c, l)
                @test t > 0
                @test all(xor.(i, o))
                @test any(o)
            end
        end

        for x in [OutlierPercentageThresholdStrategy, GroundTruthThresholdStrategy]
            @testset "$x" begin
                s = x()
                t, i, o = split_masks(s, c, l)
                @test t > 0
                @test all(xor.(i, o))
                @test any(o)
            end
        end
    end

    @testset "PWC" begin
        pwc = PWC(d, gamma)
        @test_throws ArgumentError predict(pwc, d)
        calculate_threshold!(pwc, l)
        pred = predict(pwc, d)
        @test length(pred) == size(d, 2)
        @test !all(pred .== :inlier)
    end
end
