using Knet, Dates

include("data.jl")
include("model.jl")
include("lamb.jl")
include("bertadam.jl")

function loss(model, data; average=true)    
    total_length = 0
    total_loss = 0
    for (x,y) in data
        total_length += size(x, 2)
        total_loss += model(x,y) * size(x, 2)
    end
    
    return total_loss / total_length
end

function halfdownlr(model)
    for p in params(model)
        p.opt.lr /= 2
    end
end

report_lm(loss) = floor.((loss, exp.(loss), loss ./ log(2)); digits=4)

# Uses default Adam optimizer
function trainadam!(model, trn, dev, tst...; report_iter=300)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn; lr=0.002), steps=report_iter) do y
        devloss = loss(model, dev)
        tstloss = map(d->loss(model,d), tst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        else
            halfdownlr(model)
        end
        println(stderr)
        (trn=report_lm.(tstloss), dev=report_lm(devloss))
    end
    return bestmodel
end;


# Doesn't initialize optimizer
function train!(model, dtrn, ddev, p...; report_iter=300, update_per_n_batch=1)
    losses = []
    model_name = "model_$(Int(time()÷1)).jld2"
    bestmodel = deepcopy(model)
    bestloss = 10 # loss(model, ddev)

    for p in params(model)
        p.opt.t = 0
    end
    model.new_hidden = nothing
    model.new_mems = nothing
    
    println("Total iterations = ", length(trn))
    print(Dates.format(now(), "HH:MM:SS"), "  ->  ")
    println("Dev set scores : ", report_lm(bestloss))
    flush(stdout)

    grads = []
    for (k, (x, y)) in enumerate(dtrn)
        
        # Accumulate gradients for n batches
        J = @diff model(x, y)
        if length(grads) == length(params(model))
            for (i, par) in enumerate(params(model))
                grads[i] .+= grad(J, par)
            end
        else
            for par in params(model) 
                push!(grads, grad(J, par))
            end
        end
        
        if k % update_per_n_batch == 0
            for (i, par) in enumerate(params(model))
                update!(value(par), grads[i], par.opt)
            end
            grads = []
        end

        push!(losses, value(J))
        if k % (report_iter ÷ 50) == 0
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
                println("new best dev score, saving checkpoint..\n")
                bestmodel = deepcopy(model)
                Knet.save(model_name, "model", model)
                flush(stdout)
            else
                halfdownlr(model)
            end
        end
    end
    bestmodel
end;

nothing