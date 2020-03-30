using Knet, StatsBase, LinearAlgebra, CuArrays, Random

include("data.jl")
include("layers.jl")

function loss(model, data; average=true)
    mean(model(x,y) for (x,y) in data)
end

function train!(model, trn, dev, tst...)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn), seconds=30) do y
        devloss = loss(model, dev)
        tstloss = map(d->loss(model,d), tst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (trnloss=tstloss, trnppl=exp.(tstloss), trnbpc=(tstloss ./ log(2)), devloss=devloss, devppl=exp.(devloss), devbpc=(devloss ./ log(2)) )
    end
    return bestmodel
end;