@info "Train baseline Single Headed Attention Recurrent language model using enwik8 dataset..."

using Knet

include("../src/data.jl")
include("../src/model.jl")
include("../src/train.jl")

function evaluate()
    println()
    @info "Finished training, Starting evaluation ..."
    devloss = loss(model, ddev);
    println("Development set scores:    ", report_lm(devloss))
    testloss = loss(model, dtst);
    println("Test set scores:           ", report_lm(testloss))

    model_name = "full_main.jld2"
    @info "Saving the model as $(model_name)"
    Knet.save(model_name, "model", model);
end

BATCHSIZE = 2 ; @show BATCHSIZE
BPTT = 1024 ; @show BPTT
MEMSIZE = 5000 ; @show MEMSIZE
EMSIZE = 1024 ; @show EMSIZE

datadir = "../data/enwik8"
jld2dir = "../jld2/enwik8.jld2"
if !isfile(jld2dir)
    println("Reading data from directory: $datadir")
    println("Setting batch size to $BATCHSIZE")
    vocab = Vocab("$datadir/train.txt")
    trainfile = TextReader("$datadir/train.txt", vocab)
    validfile = TextReader("$datadir/valid.txt", vocab)
    testfile = TextReader("$datadir/test.txt", vocab)
    dtrn = TextData(trainfile, batchsize=BATCHSIZE, bptt=BPTT)
    ddev = TextData(validfile, batchsize=BATCHSIZE, bptt=BPTT, randomize = false)
    dtst = TextData(testfile, batchsize=BATCHSIZE, bptt=BPTT, randomize = false)
    println("Saving data to $jld2dir")
    Knet.save(jld2dir, "dtrn", dtrn, "dtst", dtst, "ddev", ddev)
else 
    println("Loading data from $jld2dir")
    (dtrn, dtst, ddev) = Knet.load(jld2dir, "dtrn", "dtst", "ddev")
    vocab = dtrn.src.vocab
    if dtrn.batchsize != BATCHSIZE
        changebatchsize!(dtrn, BATCHSIZE)
        changebatchsize!(ddev, BATCHSIZE)
        changebatchsize!(dtst, BATCHSIZE)
    end;
    dtrn.bptt = BPTT
    dtst.bptt = BPTT
    ddev.bptt = BPTT
end;

println()
@info "Initializing the model and collecting training data..."
epochs, em_size, hidden_size, layers = 8, EMSIZE, (EMSIZE*4), 4
println("embedding size: ", em_size)
println("hidden size: ", hidden_size)
println("layers: ", layers)
println("Collecting training data...")
println("epochs: ", epochs)

ctrn = collect(dtrn)
trn = collect(flatten(collect(dtrn) for i in 1:epochs))
dev = collect(ddev)
mintrn = ctrn[1:20];

model = SHARNN(em_size, hidden_size, vocab, layers; num_max_positions=MEMSIZE);

println()
@info "Starting training, total iteration no: $(length(trn))"
initlamb!(model, length(trn); lr=0.002, warmup=(1200)/length(trn))

#  TODO: stop training at anytime using CTRL-C
model = train!(model, trn, dev, mintrn; report_iter=length(ctrn), update_per_n_batch=8)

halfdownlr(model)
model = train!(model, trn, dev, mintrn; report_iter=length(ctrn), update_per_n_batch=8)

atexit(evaluate)