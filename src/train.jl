using Knet, Dates

include("data.jl")
include("model.jl")
include("lamb.jl")
include("bertadam.jl")

function loss(model, data; average=true)
    Knet.mean([model(x,y) for (x,y) in data])
end

report_lm(loss) = floor.((loss, exp.(loss), loss ./ log(2)); digits=4)

# Uses default Adam optimizer
function trainadam!(model, trn, dev, tst...; report_iter=300)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn; lr=0.002), steps=report_iter) do y
        devloss = loss(model, dev)
        tstloss = map(d->loss(model,d), tst)
        for p in params(model)
            p.opt.lr /= (10/11)
        end
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (trn=report_lm.(tstloss), dev=report_lm(devloss))
    end
    return bestmodel
end;


# Doesn't initialize optimizer
function train!(model, dtrn, ddev, p...; report_iter=300)
    losses = []
    model_name = "model_$(Int(time()÷1)).jld2"
    bestmodel = deepcopy(model)
#     bestloss = loss(model, ddev)
    bestloss = 1e5
    println("Total iterations = ", length(trn))
    print(Dates.format(now(), "HH:MM:SS"), "  ->  ")
    println("Dev set scores : ", report_lm(bestloss))
    flush(stdout)

    for (k, (x, y)) in enumerate(dtrn)
        J = @diff model(x, y)
        for par in params(model)
            g = grad(J, par)
            update!(value(par), g, par.opt)
        end

        push!(losses, value(J))
        if k % (report_iter ÷ 100) == 0
            print(Dates.format(now(), "HH:MM:SS"), "  ->  ")
            println("$k iter: Train scores (loss, ppl, bpc): ", report_lm(Knet.mean(losses)))
            losses = []
            flush(stdout)
        end
        
        if k % report_iter == 0

            dloss = loss(model, ddev)
            print("\n", Dates.format(now(), "HH:MM:SS"), "  ->  ")
            println("$k iter: =Dev scores= (loss, ppl, bpc): ", report_lm(dloss))
            flush(stdout)
            if dloss < bestloss
                bestloss = dloss
                print(Dates.format(now(), "HH:MM:SS"), "  ->  ")
                println("new best dev score, saving checkpoint..")
                bestmodel = deepcopy(model)
                Knet.save(model_name, "model", model)
                flush(stdout)
            end
        end
    end
    bestmodel
end;

nothing