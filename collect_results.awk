BEGIN {
    sep = ","
    print "model" sep "epoch" sep "acc" sep "auroc" sep "f1" sep "hyperparams"
}

/build_transformer/ {model = "transformer"}
/build_rnn/ {model = "rnn"}
/build_cnn/ {model = "cnn"}

/build_/ {
    params = $0
    gsub(sep, ";", params)
}

/after epoch/ {
    sub(/:/, "", $(NF - 1))
    epoch = $(NF - 1)
}

/[Vv]alidation accuracy/ {
    sub(/%/, "", $(NF))
    acc[epoch] = $(NF)
}

/[Vv]alidation ROC AUC/ {
    roc[epoch] = $(NF)
}

/[Vv]alidation F1-score/ {
    f1[epoch] = $(NF)
}

/image to/ {
    for (ep in acc) {
        print model sep ep sep acc[ep] sep roc[ep] sep f1[ep] sep params
    }
    delete acc
    delete roc
    delete f1
    params = ""
}
