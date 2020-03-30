using Knet, StatsBase, LinearAlgebra, CuArrays, Random

include("data.jl")
include("layers.jl")

struct XModel
    embed::Embed
    rnn::RNN
    boom::Boom
    projection::Linear
    dropout::Real
    vocab::Vocab 
end


"""
    
    XModel

    Experimental Model

"""
function XModel(embsz::Int, hidden::Int, vocab::Vocab; layers=1, dropout=0)
    
    embed = Embed(length(vocab.i2v), embsz)
    rnn = RNN(embsz, hidden; bidirectional=false, numLayers=layers, dropout=dropout)
    boom = Boom(hidden, dim_feedforward=4hidden, shortcut=true)
    projection = Linear(hidden, length(vocab.i2v))
    
    XModel(embed, rnn, boom, projection, dropout, vocab)
end


function (s::XModel)(src, tgt; average=true)
    s.rnn.h, s.rnn.c = value(s.rnn.h), value(s.rnn.c) 
    embed = s.embed(src)
    rnn_out = s.rnn(embed)
    dims = size(rnn_out)
    rnn_out = reshape(rnn_out, dims[1], dims[2] * dims[3])
    output = s.projection(dropout( rnn_out .+ s.boom(rnn_out) , s.dropout))
    scores = reshape(output, size(output, 1), dims[2], dims[3])
    nll(scores, tgt; dims=1, average=average)
end