using Knet

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
    l.w * x .+ l.b
end


# # No need for this implementation anymore, since I added it as primitive operation to Knet
# """
#     gelu(x)
    
#     Gaussian Error Linear Unit Activation Function
#     https://arxiv.org/abs/1606.08415

#     returns: 
#         x * sigm(1.702 * x)
# """
# function gelu(x)
#     return x * sigm(1.702 * x)
# end


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
    heads::Int
    hiddensize::Int
    dropout::Real
    vq_collapsed::Bool
end

function Attention(nhid; q=true, k=false, v=false, heads=1, dropout=0)
    @assert nhid % heads == 0 "Heads must divide vector evenly"
    
    q = ifelse(q, Linear(nhid, nhid), false)
    k = ifelse(k, Linear(nhid, nhid), false)
    v = ifelse(v, Linear(nhid, nhid), false)
    qs = param0(1, 1, nhid)
    ks = param0(1, 1, nhid)
    vs = param0(1, 1, nhid)
    vq = Overparam(nhid)

    Overparam(q, k, v, qs, ks, vs, vq, heads, nhid, dropout, false)
end

"""
bmm(A, B ; transA=false, transB=false)

Perform a batch matrix-matrix product of matrices stored in A and B. size(A,2) ==
size(B,1) and size(A)[3:end] and size(B)[3:end] must match. If A is a (m,n,b...)
tensor, B is a (n,k,b...) tensor, and the output is a (m,k,b...) tensor.
"""

function attention(q, k, v, dropout)
    # q size = (batch, heads, seqlen, hid)
    (bsz, heads, qlen, dim) = size(q)
    klen = size(k, 3)
    # attention_scores = torch.matmul(query, key.transpose(-1, -2).contiguous()) / math.sqrt(dim)
end

function Attention(q, k, v,)

end
# def forward(self, query, key, value, attn_mask=None, batch_first=False, **kwargs):
#     qs, ks, vs = torch.sigmoid(self.qs), torch.sigmoid(self.ks), torch.sigmoid(self.vs)
#     if self.vq:
#         vs = self.vq(vs)
#     elif self.vq_collapsed:
#         vs = self.vs
#     if self.q:
#         query = self.q(query)
#         query = self.qln(query.float())
#     if self.k: key = self.k(key)
#     if self.v: value = self.v(value)
#     q, k, v = qs * query, ks * key, vs * value
#     if self.drop:
#         q, k, v = self.drop(q), k, self.drop(v)

#     original_q = q

#     if not batch_first:
#         q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

#     batch_size, query_len, nhid = q.size()
#     assert nhid == self.nhid
#     key_len = k.size(1)
#     ###
#     dim = self.nhid // self.heads
#     q = q.view(batch_size, query_len, self.heads, dim).transpose(1, 2)
#     k, v = [vec.view(batch_size, key_len, self.heads, dim).transpose(1, 2) for vec in [k, v]]

#     mix, focus = attention(q, k, v, dropout=self.drop, attn_mask=attn_mask, **kwargs)
#     mix = mix.transpose(1, 2).contiguous().view(batch_size, -1, self.nhid)
#     if not batch_first:
#         mix = mix.transpose(0, 1)

#     if self.r:
#         r = torch.cat([mix, original_q], dim=-1)
#         if self.drop: r = self.drop(r)
#         r = self.gelu(self.r(r))
#         mix = torch.sigmoid(self.r_gate) * mix + r

#     return mix, focus



struct Block
    
end

nothing