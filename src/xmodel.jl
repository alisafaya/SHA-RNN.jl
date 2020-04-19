using Knet, StatsBase, LinearAlgebra, Random

include("data.jl")
include("model.jl")

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
function XModel(embsz::Int, hidden::Int, vocab::Vocab; layers=1, dropout=0, shortcut=false)

    embed = Embed(length(vocab.i2v), embsz)
    rnn = RNN(embsz, hidden; bidirectional=false, numLayers=layers, dropout=dropout)
    boom = Boom(hidden, dim_feedforward=4hidden, shortcut=shortcut)
    projection = Linear(hidden, length(vocab.i2v))
    
    XModel(embed, rnn, boom, projection, dropout, vocab)
end

function (s::XModel)(src, tgt; average=true)
    s.rnn.h, s.rnn.c = value(s.rnn.h), value(s.rnn.c)
    embed = dropout(s.embed(src), s.dropout)
    rnn_out = s.rnn(embed)
    dims = size(rnn_out)
    rnn_out = reshape(rnn_out, dims[1], dims[2] * dims[3])
    output = s.projection(dropout( rnn_out .+ s.boom(rnn_out) , s.dropout))
    scores = reshape(output, size(output, 1), dims[2], dims[3])
    nll(scores, tgt; dims=1, average=average)
end;

function generate(s::XModel; start="", maxlength=30)
    s.rnn.h, s.rnn.c = 0, 0
    chars = fill(s.vocab.eos, 1)
    start = [ c for c in start ]
    starting_index = 1
    for i in 1:length(start)
        push!(chars, s.vocab.v2i[start[i]])
        charembed = s.embed(chars[i:i])
        rnn_out = s.rnn(charembed)
        starting_index += 1
    end
    
    for i in starting_index:maxlength
        charembed = s.embed(chars[i:i])
        rnn_out = s.rnn(charembed)
        output = s.projection(dropout( rnn_out .+ s.boom(rnn_out) , s.dropout))
        push!(chars, s.vocab.v2i[ sample(s.vocab.i2v, Weights(Array(softmax(reshape(output, length(s.vocab.i2v)))))) ] )
        
        if chars[end] == s.vocab.eos
            break
        end
    end
    
    join([ s.vocab.i2v[i] for i in chars ], "")
end

nothing