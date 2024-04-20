for k in 10 8 25 100 6 5 4 3 2 7 9; do
    cmd="python3 run.py -i data/etymology_top${k}.csv $*"
    for embedding_size in 256 512; do
        # rnn
        for rnn_type in rnn gru lstm; do
            for num_layers in 1 3 6 10; do
                $cmd --embedding_size "$embedding_size" rnn --rnn_type "$rnn_type" --num_layers "$num_layers"
            done
        done

        # cnn
        for num_layers in 1 3 6 10; do
            for conv_filter_count in 2 8 16; do
                for conv_kernel_size in 2 3; do
                    $cmd --embedding_size "$embedding_size" cnn --num_layers "$num_layers" --conv_filter_count "$conv_filter_count" --conv_kernel_size "$conv_kernel_size"
                done
            done
        done

        # transformer
        for num_layers in 1 3 6 10; do
            for nhead in 1 4 8 16; do
                $cmd --embedding_size "$embedding_size" transformer --num_layers "$num_layers" --nhead "$nhead"
            done
        done
    done
done
