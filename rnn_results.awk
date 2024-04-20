BEGIN {
    FS = ","
    sep = ";"

    print "rnn_type,embedding_size,num_layers,epoch,acc,auroc,f1"
}

{
    gsub(/ /, "", $0)
}

/^rnn/ {
    epoch = $2
    acc = $3
    auroc = $4
    f1 = $5
    split($6, params, sep)
    for (i in params) {
        gsub(/'/, "", params[i])
        split(params[i], kv, ":")
        opts[kv[1]] = kv[2]
    }
    print opts["rnn_type"] "," opts["embedding_size"] "," opts["num_layers"] "," epoch "," acc "," auroc "," f1
}
