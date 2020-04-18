using Knet, Dates

include("data.jl")
include("layers.jl")
include("optimizer.jl")

function loss(model, data; average=true)
    Knet.mean([model(x,y) for (x,y) in data])
end

report_lm(loss) = (loss=loss, ppl=exp.(loss), bpc=loss ./ log(2))

# Uses default Adam optimizer
function train!(model, steps, trn, dev, tst...)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn), steps=steps) do y
        devloss = loss(model, dev)
        tstloss = map(d->loss(model,d), tst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (trn=report_lm(tstloss), dev=report_lm(devloss))
    end
    return bestmodel
end;

function initopt!(model, t_total; lr=0.001, warmup=0.1)
    for par in params(model)
        if length(size(value(par))) === 1
            par.opt = BertAdam(lr=lr, warmup=warmup, t_total=t_total, w_decay_rate=0.01)
        else
            par.opt = BertAdam(lr=lr, warmup=warmup, t_total=t_total)
        end
    end
end

# # Uses default Bert Adam optimizer
# function train!(model, dtrn, ddev; report_iter=300)
#     losses = []
#     model_name = "model_$(Int(time()÷1)).jld2"
#     bestmodel = deepcopy(model)
#     bestloss = loss(model, ddev)
#     print(Dates.format(now(), "HH:MM:SS"), "  ->  ")
#     println("Dev set scores : ", report_lm(bestloss))
#     flush(stdout)
#     for (k, (x, y)) in enumerate(dtrn)
#         J = @diff model(x, y)
#         for par in params(model)
#             g = grad(J, par)
#             update!(value(par), g, par.opt)
#         end
#         push!(losses, value(J))
#         if k % report_iter == 0
#             print(Dates.format(now(), "HH:MM:SS"), "  ->  ")
#             println("$k iteration: Training set scores : ", report_lm(Knet.mean(losses)))
#             losses = []
#             flush(stdout)
#             dloss = loss(model, ddev)
#             print(Dates.format(now(), "HH:MM:SS"), "  ->  ")
#             println("Dev set scores after $k iteration : ", report_lm(dloss))
#             flush(stdout)
#             if dloss < bestloss
#                 bestloss = dloss
#                 print(Dates.format(now(), "HH:MM:SS"), "  ->  ")
#                 println("new best dev score, saving checkpoint..")
#                 bestmodel = deepcopy(model)
#                 Knet.save(model_name, "model", model)
#                 flush(stdout)
#             end
#         end
#     end
#     bestmodel
# end;

nothing