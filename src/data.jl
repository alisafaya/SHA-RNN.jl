using Base.Iterators, IterTools

struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
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
        map(w -> cdict[w] = get!(cdict, w, 0) + 1, tokens)
    end
    
    # select words with frequency higher than mincount
    # sort by frequency and delete if vocabsize is determined
    fsorted = sort([ (w, c) for (w, c) in cdict if c >= mincount ], by = x -> x[2], rev = true)
    
    vocabsize == Inf || (fsorted = fsorted[1:vocabsize])

    i2w = [ eos; unk; [ x[1] for x in fsorted[3:end] ] ]
    w2i = Dict( w => i for (i, w) in enumerate(i2w))                
    
    return Vocab(w2i, i2w, w2i[unk], w2i[eos], tokenizer)
end
                
struct TextReader
    file::String
    vocab::Vocab
end
                
function Base.iterate(r::TextReader, s=nothing)
    s === nothing && (s = open(r.file))
    eof(s) && return close(s)
    return [ get(r.vocab.w2i, w, r.vocab.unk) for w in r.vocab.tokenizer(readline(s))], s
end
                
Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

function WordsData(src::TextReader; batchsize = 128, maxlength = typemax(Int),
                batchmajor = false, bucketwidth = 2, numbuckets = min(128, maxlength ÷ bucketwidth))
    buckets = [ [] for i in 1:numbuckets ] # buckets[i] is an array of sentence pairs with similar length
    WordsData(src, batchsize, maxlength, batchmajor, bucketwidth, buckets)
end

Base.IteratorSize(::Type{WordsData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{WordsData}) = Base.HasEltype()
Base.eltype(::Type{WordsData}) = Array{Any,1}

function Base.iterate(d::WordsData, state=nothing)
    if state == 0 # When file is finished but buckets are partially full 
        for i in 1:length(d.buckets)
            if length(d.buckets[i]) > 0
                buc = d.buckets[i]
                d.buckets[i] = []
                return buc, state
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

        i = Int(ceil(src_length / d.bucketwidth))
        i > length(d.buckets) && (i = length(d.buckets))

        push!(d.buckets[i], src_word)
        if length(d.buckets[i]) == d.batchsize
            buc = d.buckets[i]
            d.buckets[i] = []
            return buc, state
        end
    end
end

function arraybatch(d::WordsData, bucket)
    src_eow = d.src.charset.eow
    src_lengths = map(x -> length(x), bucket)
    max_length = max(src_lengths...)
    x = zeros(Int64, length(bucket), d.maxlength + 1) # default d.batchmajor is false

    for (i, v) in enumerate(bucket)
        to_be_added = fill(src_eow, d.maxlength - length(v))
        x[i,:] = [src_eow; v; to_be_added]
    end
    
    d.batchmajor && (x = x')
    return (x[:, 1:end-1], x[:, 2:end]) # to calculate nll on generators output directly
end

function readwordset(fname)
    words = []
    fi = open(fname)
    while !eof(fi)
        push!(words, readline(fi))
    end
    close(fi)
    words
end