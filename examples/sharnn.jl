@info "Train baseline Single Headed Attention Recurrent language model using enwik8 dataset..."

using Knet

include("../src/data.jl")
include("../src/model.jl")
include("../src/train.jl")

BATCHSIZE = 4 ; @show BATCHSIZE
BPTT = 1024 ; @show BPTT
MEMSIZE = 2048 ; @show MEMSIZE
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
    ddev = TextData(validfile, batchsize=BATCHSIZE, bptt=BPTT)
    dtst = TextData(testfile, batchsize=BATCHSIZE, bptt=BPTT)
    println("Saving data from $jld2dir")
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
epochs, em_size, hidden_size, layers = 2, EMSIZE, (EMSIZE*4), 4
println("embedding size: ", em_size)
println("hidden size: ", hidden_size)
println("layers: ", layers)
println("Collecting training data...")
println("epochs: ", epochs)

ctrn = collect(dtrn)
trn = collect(flatten(ctrn for i in 1:epochs))
dev = collect(ddev);

# model = SHARNN(em_size, hidden_size, vocab, layers; num_max_positions=MEMSIZE);
# model = Knet.load("sharnn_$(em_size)_$(layers).jld2", "model")
model = Knet.load("model_1588696599.jld2", "model")
println()
@info "Starting training, total iteration no: $(length(trn))"
# initopt!(model, length(trn); lr=0.0005, warmup=(1000)/length(trn))
model = train!(model, trn, dev; report_iter=length(ctrn)) #  TODO: stop training at anytime using CTRL-C -> not yet :/

function evaluate()
    println()
    @info "Finished training, Starting evaluation ..."
    devloss = loss(model, ddev);
    println("Development set scores:    ", report_lm(devloss))
    testloss = loss(model, dtst);
    println("Test set scores:           ", report_lm(testloss))
    trnloss = loss(model, dtrn);
    println("Training set scores:       ", report_lm(trnloss))
    
    # @info "Generate text using the trained model"
    # print(generate(model, start="United Nations ", maxlength=1024))

    model_name = "sharnn_$(em_size)_$(layers).jld2"
    @info "Saving the model as $(model_name)"
    Knet.save(model_name, "model", model);
end

atexit(evaluate)