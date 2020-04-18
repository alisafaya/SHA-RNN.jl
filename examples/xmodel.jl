# use Knet dev version
using Knet

include("../src/data.jl")
include("../src/xmodel.jl")
include("../src/train.jl")

datadir = "../data/enwik8"
jld2dir = "../jld2/enwik8.jld2"
BATCHSIZE = 24

if !isfile(jld2dir)
    println("Reading data from directory: $datadir")
    println("Setting batch size to $BATCHSIZE")
    vocab = Vocab("$datadir/train.txt")
    trainfile = TextReader("$datadir/train.txt", vocab)
    validfile = TextReader("$datadir/valid.txt", vocab)
    testfile = TextReader("$datadir/test.txt", vocab)
    dtrn = TextData(trainfile, batchsize=BATCHSIZE)
    ddev = TextData(validfile, batchsize=BATCHSIZE)
    dtst = TextData(testfile, batchsize=BATCHSIZE)
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
end;

@info "Loading the model and collecting training data..."
epochs, em_size, hidden_size, layers = 15, 512, 512, 2;
println("embedding size: ", em_size);
println("hidden size: ", hidden_size);
println("layers: ", layers);
println("epochs: ", epochs)
ctrn = collect(dtrn);
trn = collect(flatten(shuffle!(ctrn) for i in 1:epochs));
trnmini = ctrn[1:20];
dev = collect(ddev);


@info "Starting training, total iteration no: $(length(trn))"
# model = Knet.load("../jld2/xmodel.jld2", "model")
model = XModel(em_size, hidden_size, vocab; layers=layers, dropout=0.2);
model.rnn.c, model.rnn.h = 0, 0
initopt!(model, length(trn); lr=0.003)
model = train!(model, trn, dev; report_iter=1000)


model.rnn.c, model.rnn.h = 0, 0
initopt!(model, length(trn); lr=0.001)
epochs = 5
trn = collect(flatten(shuffle!(ctrn) for i in 1:epochs));
@info "Starting tuning, total iteration no: $(length(trn))"
model = train!(model, trn, dev; report_iter=500)

@info "Finished training, Starting evaluation ..."
trnloss = loss(model, dtrn);
println("Training set scores:       ", report(trnloss))
devloss = loss(model, ddev);
println("Development set scores:    ", report(devloss))
testloss = loss(model, dtst);
println("Test set scores:           ", report(testloss))

# @info "Generate text using the trained model"
# print(generate(model, start="United Nations ", maxlength=1024))

@info "Saving the model as model.jld2"
Knet.save("model_x.jld2", "model", model);
