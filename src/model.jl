using Knet

include("data.jl")
include("layers.jl")

# def init_weights(self, module):
# if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
#     module.weight.data.normal_(mean=0.0, std=0.1 / np.sqrt(self.ninp))

# if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
#     module.bias.data.zero_()
function initweights(m::SHARNN)
    # init_weights
    m
end


struct SHARNN
    encoder::Embed
    rnn::RNN        
    decoder::Linear  
    vocab::Vocab
    blocks::Vector{Block}
    inputdrop::Real
    drop::Real
    hiddendrop::Real
end

function SHARNN(embsz::Int, hidden::Int, vocab::Vocab; layers=1, dropout=0, tie_weights=false)
    
    encoder = Embed(length(vocab.i2v), embsz)
    rnn = RNN(embsz, hidden; bidirectional=false, numLayers=layers, dropout=dropout)
    decoder = Linear(hidden, length(vocab.i2v))
    # model = SHARNN(encoder, rnn, , dropout, vocab)
    # initweights(model)
    # return model
end

