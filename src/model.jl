using Knet, LinearAlgebra, CUDA, Statistics

function Knet.KnetArray(x::CuArray{T,N}) where {T,N}
    p = Base.bitcast(Knet.Cptr, pointer(x))
    k = Knet.KnetPtr(p, sizeof(x), Int(CUDA.device().handle), x)
    KnetArray{T,N}(k, size(x))
end

aType = CUDA.functional() ? KnetArray{Float32} : Array{Float32} 
# aType = CUDA.functional() ? CuArray{Float32} : Array{Float32} 

function normal_init(a...; mean=0.0, std=0.1)
    w = randn(a...)
    if ndims(w) == 1
        fanin = length(w)
    elseif ndims(w) == 2
        fanin = size(w,2)
    else
        fanin = div(length(w),  a[end])
    end
    T = eltype(w)
    w .* T(std / sqrt(fanin)) .+ T(mean)
end
init_f = normal_init

mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))
# std2(a, μ, ϵ) = sqrt.(mean(abs2.(a .- μ), dims=1) .+ ϵ)

struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    Embed(param(embedsize, vocabsize; atype=aType, init=init_f))
end

function (l::Embed)(x)
    l.w[:, x]
end

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize, inputsize; atype=aType, init=init_f), param0(outputsize; atype=aType))
end

function (l::Linear)(x)
    mmul(l.w, x) .+ l.b
end


"""
    LayerNorm(size::Integer; eps=1e-5)
    LayerNorm(γ, β, ϵ)

References:
* [Ba, Kiros and Hinton 2016](https://arxiv.org/abs/1607.06450) Layer Normalization
* torch.nn.LayerNorm
* tf.keras.layers.LayerNormalization
"""
struct LayerNorm; γ; β; ϵ; end

function LayerNorm(dmodel; eps=1e-5)
    γ = param(dmodel; init=ones, atype=aType)
    β = param(dmodel; init=zeros, atype=aType)
    LayerNorm(γ, β, eps)
end

function (l::LayerNorm)(x, o...)
    μ = mean(x,dims=1)
    σ = std(x,mean=μ,dims=1)
    ϵ = eltype(x)(l.ϵ)
    l.γ .* (x .- μ) ./ (σ .+ ϵ) .+ l.β # TODO: doing x .- μ twice?
end

"""
    gelu(x)
    
Gaussian Error Linear Unit Activation Function
https://arxiv.org/abs/1606.08415
returns: 
    x * sigm(1.702 * x)
"""
function gelu(x)
    return x * sigm(Float32(1.702) * x)
end

struct Boom
    lin1
    lin2
    dropout::Float32
    shortcut
    act
end

function Boom(inpsize::Int; dim_feedforward=2048, dropout=0.1, shortcut=false, act=gelu)
    lin1 = Linear(inpsize, dim_feedforward)

    if shortcut
        lin2 = aType(zeros(Float32, inpsize, dim_feedforward))
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
        x = mmul(boom.lin2, x)
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
    Overparam(Linear(nhid, 2nhid), nhid)
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
    qln
    heads::Int
    hiddensize::Int
    depth::Int
    dropout::Real
end

function Attention(nhid; q=true, k=false, v=false, heads=1, dropout=0)
    @assert nhid % heads == 0 "Heads must divide vector evenly"

    q = q ? Linear(nhid, nhid) : false
    k = k ? Linear(nhid, nhid) : false
    v = v ? Linear(nhid, nhid) : false
    qs = param0(nhid, 1; atype=aType)
    ks = param0(nhid, 1; atype=aType)
    vs = param0(nhid, 1; atype=aType)
    vq = Overparam(nhid)
    qln = LayerNorm(nhid, eps=1e-12)
    depth = nhid ÷ heads

    Attention(q, k, v, qs, ks, vs, vq, qln, heads, nhid, depth, dropout)
end

function (a::Attention)(query, key, value; attn_mask=false)
    qs, ks, vs = sigm.(a.qs), sigm.(a.ks), sigm.(a.vs)
    vs = a.vq(vs)
    
    query = a.q !== false ? a.q(query) : query;
    query = a.q !== false ? a.qln(query) : query;
    key = a.k !== false ? a.k(key) : key;
    value = a.v !== false ? a.v(value) : value;

    q = qs[:] .* query
    k = ks[:] .* key
    v = vs[:] .* value

    q = permutedims(q, [3, 1, 2]) # -> B, H, T
    k = permutedims(k, [3, 1, 2]) # -> B, H, T
    v = permutedims(v, [3, 1, 2]) # -> B, H, T

    q = reshape(q, size(q, 1), a.depth, a.heads, :)
    k = reshape(k, size(k, 1), a.depth, a.heads, :)
    v = reshape(v, size(v, 1), a.depth, a.heads, :)

    mix, focus = attention(q, k, v; attn_mask=attn_mask, adropout=a.dropout) # mix -> (Tv, depth, heads, B)

    mix = reshape(mix, size(mix, 1), :, size(mix, 4))
    mix = permutedims(mix, (2, 3, 1))

    mix, focus # mix -> H, B, T
end

function attention(q, k, v; attn_mask=nothing, adropout=0.0)
    qk = bmm(q, k, transB=true)
    attn_scores = qk ./ Float32(sqrt(size(q, 1)))

    # to be checked
    if attn_mask !== false
        # attn_scores .+= attn_mask # error on backward pass, -> no method matching copyto!(::KnetArray{Float32,4}, ::Base.Broadcast.Broadcasted{Base.Broadcast.Style{AutoGrad.Value},NTuple{4,Base.OneTo{Int64}},typeof(identity),Tuple{AutoGrad.Result{KnetArray{Float32,4}}}})
        attn_scores = attn_mask .+ attn_scores
    end

    attn_scores = softmax(attn_scores, dims=2) # on K dim
    
    if adropout != 0
        dropout(attn_scores, adropout) # attention dropout
    end

    return bmm(attn_scores, v), attn_scores
end

struct SHARNNBlock
    attn
    ff
    dropout
    residual
    rnn
    lnstart
    lnmid
    lnmem
    lnff
    lnxff
end

function SHARNNBlock(embed_dim, hidden_dim; heads=1, dropout=0.0, residual=true, use_attn=true)
    
    if use_attn
        attn = Attention(embed_dim; heads=heads, dropout=dropout)
        lnmid = LayerNorm(embed_dim, eps=1e-12)
        lnmem = LayerNorm(embed_dim, eps=1e-12)
    else
        attn = nothing
        lnmid = nothing
        lnmem = nothing
    end

    ff = Boom(embed_dim; dim_feedforward=hidden_dim, dropout=dropout, shortcut=true, act=gelu)
    rnn = RNN(embed_dim, embed_dim, atype=aType)
    lnstart = LayerNorm(embed_dim, eps=1e-12)
    lnff = LayerNorm(embed_dim, eps=1e-12)
    lnxff = LayerNorm(embed_dim, eps=1e-12)

    SHARNNBlock(attn, ff, dropout, residual, rnn, lnstart, lnmid, lnmem, lnff, lnxff)
end

function (b::SHARNNBlock)(h, p_encoding, attn_mask; mem=nothing, hidden=nothing)
    h = b.lnstart(h); 

    # RNN part
    if hidden !== nothing
        b.rnn.h, b.rnn.c = hidden
    else
        b.rnn.h, b.rnn.c = 0, 0
    end

    x = b.rnn(h)
    x = dropout(x, b.dropout)
    new_hidden = (value(b.rnn.h), value(b.rnn.c))
    h = b.residual ? (h + x) : x  

    # Attention part
    
    focus, new_mem = nothing, []
    if b.attn !== nothing
        if mem !== nothing
            bigh = cat(mem, b.lnmem(h), dims=3)
        else
            bigh = b.lnmem(h)
        end

        newmemsize = length(p_encoding)
        newmem_start_idx = newmemsize < (size(bigh, 3) + 1) ? (size(bigh, 3) + 1 - newmemsize) : 1
        new_mem = value(bigh[:, :, newmem_start_idx:end])
        
        h = b.lnmid(h)
        x, focus = b.attn(h, bigh, bigh; attn_mask=attn_mask)
        x = dropout(x, b.dropout)
        h = x + h
    end

    # Boom layer
    h, x = b.lnff(h), b.ff(b.lnxff(h))
    x = dropout(x, b.dropout)
    h = h + x

    h, new_mem, new_hidden, focus
end

mutable struct SHARNN
    encoder
    decoder
    vocab
    blocks
    idrop
    drop
    hdrop
    num_max_positions
    pos_emb
    new_hidden
    new_mems
end

function SHARNN(embsz::Int, hidden::Int, vocab::Vocab, nlayers::Int; num_max_positions=1024, nheads=1, dropout=0.0, dropouth=0.1, dropouti=0.1, wdrop=0, tie_weights=false)
    
    encoder = Embed(length(vocab.i2v), embsz)
    
    pos_emb = zeros(num_max_positions)

    if tie_weights
        decoder = nothing
    else
        decoder = Linear(embsz, length(vocab.i2v))
    end

    blocks = []
    for i=1:nlayers
        push!(blocks, SHARNNBlock(embsz, hidden; heads=nheads, dropout=dropouth, residual=false, use_attn=(i == (nlayers - 1)))) # (i == (nlayers - 1))
    end

    SHARNN(encoder, decoder, vocab, blocks, dropouti, dropout, dropouth, num_max_positions, pos_emb, nothing, nothing)
end

function (s::SHARNN)(x; hidden=nothing, mems=nothing, return_h=true)
    embed = dropout(s.encoder(x), s.idrop)
    
    if mems !== nothing
        maxmem = s.num_max_positions - size(embed, 3)
        maxmem_start_idx = maxmem < (size(mems[1], 3) + 1) ? (size(mems[1], 3) + 1 - maxmem) : 1
        mems = [ m[:, :, maxmem_start_idx:end] for m in mems]
    end

    new_hidden, new_mems, focus = [], [], []
    attn_mask = fill(typemin(Float32), size(embed, 3), size(embed, 3)) # -> TxT
    attn_mask = triu(attn_mask, 1)
    if mems !== nothing
        max_mem_size = max([size(m, 3) for m in mems]...)
        attn_mask = cat(zeros(Float32, size(embed, 3), max_mem_size), attn_mask, dims=2)
    end
    attn_mask = aType(attn_mask)

    h = embed
    for (i, block) in enumerate(s.blocks)
        mem = mems !== nothing ? mems[i] : nothing
        hid = hidden !== nothing ? hidden[i] : nothing
        h, m, nh, f = block(h, s.pos_emb, attn_mask; mem=mem, hidden=hid)
        push!(new_hidden, nh)
        push!(new_mems, m)
    end
    h = dropout(h, s.drop)

    h, new_hidden, new_mems
end


function (s::SHARNN)(src, tgt; average=true)
    h, new_hidden, new_mems = s(src; hidden=s.new_hidden, mems=s.new_mems) 
    s.new_hidden = new_hidden
    s.new_mems = new_mems
    
    if s.decoder !== nothing
        scores = s.decoder(h)
    else
        scores = mmul(s.encoder.w', h)
    end
    
    nll(scores, tgt; dims=1, average=average)
end

nothing