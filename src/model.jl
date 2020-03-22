using Knet, StatsBase, LinearAlgebra, CuArrays, Random

include("data.jl")

struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    Embed(param(embedsize, vocabsize))
end

function (l::Embed)(x)
    l.w[:, x]
end

struct Linear; w; end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize, inputsize))
end

function (l::Linear)(x)
    l.w * x
end

"""
    Simple RNN Language model
"""
struct SimpleLSTMModel
    embed::Embed
    rnn::RNN        
    projection::Linear  
    dropout::Real
    vocab::Vocab 
end

"""
    SimpleLSTMModel(embsz::Int, hidden::Int, vocab::Vocab; layers=1, dropout=0)

embsz: Embeddings size
hidden: Size of LSTM hidden layer
vocab: Vocab set of type ::Vocab
layers=1: number of LSTM layers
dropout=0: dropout value

Returns:
    Language model consisting of LSTM with the given parameters
"""
function SimpleLSTMModel(embsz::Int, hidden::Int, vocab::Vocab; layers=1, dropout=0)
    
    embed = Embed(length(vocab.i2v), embsz)
    rnn = RNN(embsz, hidden; bidirectional=false, numLayers=layers, dropout=dropout)
    projection = Linear(hidden, length(vocab.i2v))
    
    SimpleLSTMModel(embed, rnn, projection, dropout, vocab)
end

# NOT USED
# function mask(a, pad)
#     a = copy(a)
#     for i in 1:size(a, 1)
#         j = size(a,2)
#         while a[i, j] == pad && j > 1
#             if a[i, j - 1] == pad
#                 a[i, j] = 0
#             end
#             j -= 1
#         end
#     end
#     return a
# end

""" 
    Language model loss function
"""
function (s::SimpleLSTMModel)(src, tgt; average=true)
    s.rnn.h, s.rnn.c = 0, 0
    embed = s.embed(src)
    rnn_out = s.rnn(embed)
    dims = size(rnn_out)
    output = s.projection(dropout(reshape(rnn_out, dims[1], dims[2] * dims[3]), s.dropout))
    scores = reshape(output, size(output, 1), dims[2], dims[3])
    nll(scores, tgt; dims=1, average=average)
end

"""
    Total loss
"""
function loss(model, data; average=true)
    mean(model(x,y) for (x,y) in data)
end

"""
    Generating words using the LM with sampling
"""
function generate(s::SimpleLSTMModel; start="", del="", maxlength=30)
    s.rnn.h, s.rnn.c = 0, 0
    vocabs = fill(s.vocab.eos, 1)
    
    starting_index = 1
    for (i, t) in enumerate(start)
        push!(vocabs, s.vocab.v2i[string(UInt8(t))])
        embed = s.embed(vocabs[i:i])
        rnn_out = s.rnn(embed)
        starting_index += 1
    end
    
    for i in starting_index:maxlength
        embed = s.embed(vocabs[i:i])
        rnn_out = s.rnn(embed)
        output = s.projection(dropout(rnn_out, s.dropout))
        push!(vocabs, s.vocab.v2i[ sample(s.vocab.i2v, Weights(Array(softmax(reshape(output, length(s.vocab.i2v)))))) ] )
        
        if vocabs[end] == s.vocab.eos
            break
        end
    end
    join([ Char(parse(UInt8, s.vocab.i2v[i])) for i in vocabs[2:end-1] ], del)
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