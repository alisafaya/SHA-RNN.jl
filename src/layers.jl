using Knet

mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))

struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    Embed(param(embedsize, vocabsize))
end

function (l::Embed)(x)
    l.w[:, x]
end

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize, inputsize), param0(outputsize))
end

function (l::Linear)(x)
    mmul(l.w, x)
end


"""
    gelu(x)
    
    Gaussian Error Linear Unit Activation Function
    https://arxiv.org/abs/1606.08415

    returns: 
        x * sigm(1.702 * x)
"""
function gelu(x)
    return x * sigm(1.702 * x)
end


"""
    Boom Layer
"""
struct Boom
    lin1
    lin2
    dropout
    shortcut
    act
end

function Boom(inpsize::Int; dim_feedforward=2048, dropout=0.1, shortcut=false, act=gelu)
    lin1 = Linear(inpsize, dim_feedforward)

    if shortcut
        lin2 = KnetArray(zeros(Float32, inpsize, dim_feedforward))
        for i in 1:inpsize:dim_feedforward
            lin2[ (i ÷ inpsize) + 1 , i:(i + inpsize - 1)] .= 1
        end
    else
        lin2 = Linear(dim_feedforward, inpsize)
    end

    Boom(lin1, lin2, dropout, shortcut, act)
end

function (boom::Boom)(inp)
    x = boom.act.(boom.lin1(inp))
    x = dropout(x, boom.dropout)
    
    if boom.shortcut
        # this part takes the output of the first linear layer and splits it into ( dim_feedforward ÷ inpsize ) parts and sums them together
        # Doing this using one matrix multiplication is cheaper than two reshapes and summation 
        x = boom.lin2 * x 
    else
        x = boom.lin2(x)
    end
    return x
end


struct Overparam
    linear
    nhid
end

function Overparam(nhid)
    Overparam(Linear(nhid, nnhid), nhid)
end

function (o::Overparam)(x)
    x = o.linear(x)
    tanh.(x[1:o.nhid, :]) .* sigm.(x[o.nhid+1:end, :])
end

struct Attention
    q
    k
    v
    qs
    ks
    vs
    vq
    heads::Int
    hiddensize::Int
    dropout::Real
    vq_collapsed::Bool
end

function Attention(nhid; q=true, k=false, v=false, heads=1, dropout=0)
    @assert nhid % heads == 0 "Heads must divide vector evenly"
    
    q = q ? Linear(nhid, nhid) : false
    k = k ? Linear(nhid, nhid) : false
    v = v ? Linear(nhid, nhid) : false
    qs = param0(nhid)
    ks = param0(nhid)
    vs = param0(nhid)
    vq = Overparam(nhid)

    Attention(q, k, v, qs, ks, vs, vq, heads, nhid, dropout, false)
end


function (a::Attention)(query, key, value; attn_mask=false, args...)
    qs, ks, vs = sigm.(a.qs), sigm.(a.ks), sigm.(a.vs)
    # over parameterization vs = a.vq(vs)
    
    query = a.q !== false ? a.q(query) : query # add LayerNorm. for query here
    key = a.k !== false ? a.k(key) : key
    value = a.v !== false ? a.v(value) : value

    q, k, v = qs .* query, ks .* key, vs .* value
    q, k, v = permutedims(q, [2, 1, 3]), permutedims(k,[ 2, 1, 3]), permutedims(v, [2, 1, 3])
    
    # check for batchmajor
    
    
end


# struct Block
    
# end

nothing