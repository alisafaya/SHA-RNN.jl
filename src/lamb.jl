# Min Trust LAMB (https://github.com/Smerity/pytorch-lamb/blob/master/pytorch_lamb/lamb.py)
#
# It has been proposed in Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.
# https://arxiv.org/abs/1904.00962
#
# Adapted from BertAdam https://github.com/OsmanMutlu/BERT.jl/blob/master/src/optimizer.jl 

import Knet: update!
using AutoGrad: full
using LinearAlgebra
using CUDA

warmup_cosine(x, warmup=0.002) = x < warmup ? x/warmup : 0.5 * (1.0 + cos(π * x))
warmup_constant(x, warmup=0.002) = x < warmup ? x/warmup : 1.0
warmup_linear(x, warmup=0.002) = x < warmup ? x/warmup : 1.0 - x

mutable struct MinTrustLAMB
    lr::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    eps::AbstractFloat
    w_decay::AbstractFloat
    min_trust::AbstractFloat
    gclip::AbstractFloat
    t::Int
    schedule
    warmup
    t_total
    fstm
    scndm
end

MinTrustLAMB(; min_trust=0.25, lr=1e-3, gclip=0.25, beta1=0.9, beta2=0.999, eps=1e-6, w_decay=0.0, schedule="warmup_constant", warmup=-1, t_total=-1) = MinTrustLAMB(lr, beta1, beta2, eps, w_decay, min_trust, gclip, 0, schedule, warmup, t_total, nothing, nothing)

function initlamb!(model, t_total; min_trust=0.25, lr=0.001, warmup=0.1, schedule="warmup_constant")
    for par in params(model)
        if length(size(value(par))) === 1
            par.opt = MinTrustLAMB(;min_trust=min_trust, lr=lr, warmup=warmup, schedule=schedule, t_total=t_total, w_decay=1.2e-6)
        else
            par.opt = MinTrustLAMB(;min_trust=min_trust, lr=lr, warmup=warmup, schedule=schedule, t_total=t_total)
        end
    end
end

for T in (Array{Float32},Array{Float64},KnetArray{Float32},KnetArray{Float64},CuArray{Float32},CuArray{Float64})
    @eval begin
        function update!(w::$T, g, p::MinTrustLAMB)
            Knet.Train20.gclip!(g, p.gclip)
            g = full(g)
            if p.fstm===nothing; p.fstm=zero(w); p.scndm=zero(w); end
            p.t += 1

            # Decay the first and second moment running average coefficient
            # scndm : v_t, fstm : m_t
            lmul!(p.beta1, p.fstm)
            axpy!(1-p.beta1, g, p.fstm)
            lmul!(p.beta2, p.scndm)
            axpy!(1-p.beta2, g .* g, p.scndm)

            # Get Learning rate from scheduler
            if p.t_total !== -1
                schedule_func = eval(Meta.parse(p.schedule))
                lr_scheduled = p.lr * schedule_func(p.t/p.t_total, p.warmup)
            else
                lr_scheduled = p.lr
            end
            
            # Calculate Opt. Step
            if p.w_decay > 0.0
                adam_step = (p.fstm ./ (sqrt.(p.scndm) .+ p.eps)) .+ (p.w_decay * w)
            else
                adam_step = (p.fstm ./ (sqrt.(p.scndm) .+ p.eps))
            end

            # Calculate Trust Ratio
            weight_norm = clamp(norm(w), 0, 10)
            adam_norm = norm(adam_step)
            if weight_norm == 0 || adam_norm == 0
                trust_ratio = 1
            else
                trust_ratio = weight_norm / adam_norm
            end
            trust_ratio = max(trust_ratio, p.min_trust)
            
            # Update weight matrix
            axpy!(-lr_scheduled * trust_ratio, adam_step, w)
        end
    end
end;

nothing