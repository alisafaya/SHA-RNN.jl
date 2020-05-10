# Min Trust LAMB (https://github.com/Smerity/pytorch-lamb/blob/master/pytorch_lamb/lamb.py)
#
# It has been proposed in Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.
# https://arxiv.org/abs/1904.00962
#
# Adapted from BertAdam https://github.com/OsmanMutlu/BERT.jl/blob/master/src/optimizer.jl 

import Knet: update!
using AutoGrad: full

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

MinTrustLAMB(; min_trust=0.25, lr=1e-3, gclip=1.0, beta1=0.9, beta2=0.999, eps=1e-6, w_decay_rate=0.0, schedule="warmup_constant", warmup=-1, t_total=-1) = \
    MinTrustLAMB(lr, beta1, beta2, eps, w_decay_rate, min_trust, gclip, 0, schedule, warmup, t_total, nothing, nothing)

for T in (Array{Float32},Array{Float64},KnetArray{Float32},KnetArray{Float64})
    @eval begin
        function update!(w::$T, g, p::MinTrustLAMB)
            # Knet.gclip!(g, p.gclip)
            g = full(g)
            if p.fstm===nothing; p.fstm=zero(w); p.scndm=zero(w); end
            
            p.t += 1
            # Decay the first and second moment running average coefficient
            # scndm : v_t, fstm : m_t
            
            lmul!(p.beta1, p.fstm)
            axpy!(1-p.beta1, g, p.fstm)
            lmul!(p.beta2, p.scndm)
            axpy!(1-p.beta2, g .* g, p.scndm)

            if p.t_total !== -1
                schedule_func = eval(Meta.parse(p.schedule))
                lr_scheduled = p.lr * schedule_func(p.t/p.t_total, p.warmup)
            else
                lr_scheduled = p.lr
            end
    
            # weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

            # if p.w_decay_rate > 0.0
            #     axpy!(-lr_scheduled, (p.fstm ./ (sqrt.(p.scndm) .+ p.eps)) .+ (p.w_decay_rate * w), w)
            # else
            #     axpy!(-lr_scheduled, (p.fstm ./ (sqrt.(p.scndm) .+ p.eps)), w)
            # end


            # weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

            # adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
            # if group['weight_decay'] != 0:
            #     adam_step.add_(group['weight_decay'], p.data)

            # adam_norm = adam_step.pow(2).sum().sqrt()
            # if weight_norm == 0 or adam_norm == 0:
            #     trust_ratio = 1
            # else:
            #     trust_ratio = weight_norm / adam_norm
            # if self.min_trust:
            #     trust_ratio = max(trust_ratio, self.min_trust)
            # state['weight_norm'] = weight_norm
            # state['adam_norm'] = adam_norm
            # state['trust_ratio'] = trust_ratio
            # if self.adam:
            #     trust_ratio = 1

            # p.data.add_(-step_size * trust_ratio, adam_step)


        end
    end
end;

nothing