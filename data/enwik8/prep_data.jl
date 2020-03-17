# download and extract data if data/ directory does not exist
if !isfile("enwik8")
    enwik8 = download("http://www.mattmahoney.net/dc/enwik8.zip");
    run(`unzip $enwik8`)
    rm(enwik8)
end

if isfile("train.txt")
    println("Tokenized enwik8 already exists - skipping processing")
else
    f = open("enwik8", "r");
    data = Array{UInt8, 1}()
    readbytes!(f, data, typemax(Int))
    println("Length of enwik8: ", length(data));
    num_test_chars = 5000000;

    train_data = data[1:length(data) - (2 * num_test_chars)];
    valid_data = data[length(train_data) + 1: length(train_data) + num_test_chars];
    test_data = data[length(train_data) + num_test_chars + 1:end];

    for (fn, part) in [("train.txt", train_data), ("valid.txt", valid_data), ("test.txt", test_data)]
        println(fn, " will have ", length(part), " bytes")
        println("- Writing ", part, " ...")
        write(fn * ".raw", part)
        part_str = join(part, " ");
        open(fn, "w") do f
            write(f, part_str)
        end;
        # f = open(fn + '.raw', 'wb').write(part)
    end
end;
