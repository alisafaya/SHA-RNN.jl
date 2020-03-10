using Base.Iterators, IterTools

struct Vocab
    v2i::Dict{String,Int}
    i2v::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    # set unk and eos tokens frequency to inf because
    # we don't want them to be removed from the vocab set
    cdict = Dict(eos => Inf, unk=>Inf) 
    
    # create vocab set and count occurrences
    for l in eachline(file)
        tokens = tokenizer(l)
        map(v -> cdict[v] = get!(cdict, v, 0) + 1, tokens)
    end
    
    # select vocabs with frequency higher than mincount
    # sort by frequency and delete if vocabsize is determined
    fsorted = sort([ (v, c) for (v, c) in cdict if c >= mincount ], by = x -> x[2], rev = true)
    
    vocabsize == Inf || (fsorted = fsorted[1:vocabsize])

    i2v = [ eos; unk; [ x[1] for x in fsorted[3:end] ] ]
    v2i = Dict( v => i for (i, v) in enumerate(i2v))                
    
    return Vocab(v2i, i2v, v2i[unk], v2i[eos], tokenizer)
end
                
struct TextReader
    file::String
    vocab::Vocab
end
                
function Base.iterate(r::TextReader, s=nothing)
    s === nothing && (s = open(r.file))
    eof(s) && return close(s)
    return [ get(r.vocab.v2i, v, r.vocab.unk) for v in r.vocab.tokenizer(readline(s))], s
end
                
Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

struct VocabData
    src::TextReader        # reader for text data
    batchsize::Int         # desired batch size
    maxlength::Int         # skip if source sentence above maxlength
    batchmajor::Bool       # batch dims (B,T) if batchmajor=false (default) or (T,B) if true.
    bucketwidth::Int       # batch sentences with length within bucketwidth of each other
    buckets::Vector        # sentences collected in separate arrays called buckets for each length range
    batchmaker::Function   # function that turns a bucket into a batch.
end

function VocabData(src::TextReader; batchsize = 128, maxlength = typemax(Int),
                batchmajor = false, bucketwidth = 2, numbuckets = min(128, maxlength ÷ bucketwidth), batchmaker=arraybatch)
    buckets = [ [] for i in 1:numbuckets ] # buckets[i] is an array of sentence pairs with similar length
    VocabData(src, batchsize, maxlength, batchmajor, bucketwidth, buckets, batchmaker)
end

Base.IteratorSize(::Type{VocabData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{VocabData}) = Base.HasEltype()
Base.eltype(::Type{VocabData}) = Tuple{Array{Int64,2},Array{Int64,2}}

function Base.iterate(d::VocabData, state=nothing)
    if state == 0 # When file is finished but buckets are not empty yet 
        for i in 1:length(d.buckets)
            if length(d.buckets[i]) > 0
                batch = d.batchmaker(d, d.buckets[i])
                d.buckets[i] = []
                return batch, state
            end
        end
        return nothing # Finish iteration
    end

    while true
        src_next = iterate(d.src, state)
        
        if src_next === nothing
            state = 0
            return iterate(d, state)
        end
        
        (src_word, src_state) = src_next
        state = src_state
        src_length = length(src_word)
        
        (src_length > d.maxlength) && continue
        (src_length < 1) && continue

        i = Int(ceil(src_length / d.bucketwidth))
        i > length(d.buckets) && (i = length(d.buckets))

        push!(d.buckets[i], src_word)
        if length(d.buckets[i]) == d.batchsize
            batch = d.batchmaker(d, d.buckets[i])
            d.buckets[i] = []
            return batch, state
        end
    end
end

function arraybatch(d::VocabData, bucket)
    src_eos = d.src.vocab.eos
    src_lengths = map(x -> length(x), bucket)
    max_length = max(src_lengths...)
    x = zeros(Int64, length(bucket), d.maxlength + 1) # default d.batchmajor is false

    for (i, v) in enumerate(bucket)
        to_be_added = fill(src_eos, d.maxlength - length(v))
        x[i,:] = [src_eos; v; to_be_added]
    end
    
    d.batchmajor && (x = x')
    return (x[:, 1:end-1], x[:, 2:end])
end

# Utility to convert int arrays to sentence strings
function int2string(y, vocab::Vocab)
    y = vec(y)
    ysos = findnext(w->!isequal(w, vocab.eos), y, 1)
    ysos === nothing && return ""
    yeos = something(findnext(isequal(vocab.eos), y, ysos), 1+length(y))
    join(vocab.i2v[y[ysos:yeos-1]], " ")
end