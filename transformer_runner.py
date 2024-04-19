from math import ceil

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

from transformer_model import Transformer
from dataloader import EtymologyDataLoader, EtymologyDataset, df_to_array

from model.cnn import EtymologyCNN
from model.rnn import EtymologyRNN
from model.transformer_encoder import EtymologyTransformerEncoder

DATA = r"data/etymology_top10.csv"

if __name__ == "__main__":
    if gpu := torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")

    df = pd.read_csv(DATA)
    input_vec, target_vec, i_to_lang, i_to_char, vocab_size, num_output_classes = df_to_array(df)
    padding_idx = vocab_size - 1

    X_train, X_test, y_train, y_test = train_test_split(input_vec, target_vec, test_size=0.4)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    # Hyperparameters
    word_embedding_size = 512
    batch_size = 256
    num_epochs = 50
    learning_rate = 1.0

    model = EtymologyCNN(
        vocab_size=vocab_size,
        embedding_size=word_embedding_size,
        num_classes=num_output_classes,
        conv_layers=(
            (3, 1, 1, 2, 2, 0),
        ),
        conv_filter_count=16,
        padding_idx=padding_idx,
    )

    # model = Transformer(
    #     src_vocab_size=vocab_size,
    #     d_model=word_embedding_size,
    #     d_ff=2048,
    #     num_heads=4,
    #     output_classes=num_output_classes,
    #     num_layers=8,
    #     padding_idx=vocab_size - 1
    # )

    # model = EtymologyRNN(
    #     vocab_size=vocab_size,
    #     embedding_size=word_embedding_size,
    #     hidden_size=word_embedding_size,
    #     num_classes=num_output_classes,
    #     padding_idx=padding_idx,
    # )

    # model = EtymologyTransformerEncoder(
    #     vocab_size=vocab_size,
    #     num_classes=num_output_classes,
    #     embedding_size=word_embedding_size,
    #     padding_idx=padding_idx,
    # )

    train_dataset = EtymologyDataset(X_train, y_train)
    train_dl = EtymologyDataLoader(train_dataset, batch_size=batch_size, pin_memory=gpu)

    val_dataset = EtymologyDataset(X_val, y_val)
    val_dl = EtymologyDataLoader(val_dataset, batch_size=batch_size, pin_memory=gpu)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab_size - 1)

    model.to(device)
    criterion.to(device)

    accumulation_steps = 8
    num_training_steps = num_epochs * len(train_dl)
    num_optimizer_steps = ceil(num_training_steps / accumulation_steps)
    num_warmup_steps = num_optimizer_steps // 8
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: word_embedding_size ** (-0.5) * min((step if step else 1) ** (-0.5), (step if step else 1) * 300 ** (-1.5)),
    )

    validate_freq = 5

    # scaler = GradScaler()

    # DATA
    learning_rate = []
    validation_score = []
    loss_data = []

    model.train()
    barfmt = ('{l_bar}{bar}| %d/' + str(num_epochs)
              + ' [{elapsed}<{remaining}{postfix}]')
    with tqdm(total=num_training_steps, desc='Training', unit='epoch',
              bar_format=barfmt % 0, position=0, dynamic_ncols=True) as pbar:
        for epoch in trange(1, num_epochs + 1):
            with tqdm(total=len(train_dl), desc=f'Epoch {epoch}', leave=False, unit='batch',
                      position=1, dynamic_ncols=True) as it:

                # train
                model.train()
                for i, (X, y) in enumerate(train_dl):
                    X = X.to(device)
                    y = y.to(device).type(torch.long)

                    # with autocast():
                    logits = model(X)
                    loss = criterion(logits, y) / accumulation_steps

                    learning_rate.append(lr_scheduler.get_last_lr())
                    loss_data.append(loss.item())
                    loss.backward()
                    # scaler.scale(loss).backward()
                    if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(train_dl)):
                        # scaler.step(optimizer)
                        # scaler.update()
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    it.set_postfix(loss=loss.item() * accumulation_steps)
                    it.update()
                    pbar.update()

            # eval
            if epoch % validate_freq == 0:
                model.eval()
                accurate = 0
                with tqdm(total=len(val_dl), desc=f'Validating Epoch {epoch}', leave=False, unit='batch',
                          position=1, dynamic_ncols=True) as it:
                    for i, (X, y) in enumerate(val_dl):
                        X = X.to(device)
                        y = y.to(device)

                        with torch.no_grad():
                            logits = model.forward(X)
                            output = torch.argmax(logits, dim=-1)
                            accurate += sum(output == y)

                            it.update()

                accuracy = accurate / len(y_val)
                tqdm.write(f"Validation accuracy after epoch {epoch}: {accuracy:.4%}")
                validation_score.append(accuracy.cpu())

            pbar.bar_format = barfmt % epoch

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.8, 9.6))
    save = True

    training_steps = [x for x in range(1, num_training_steps + 1)]

    ax1.plot(training_steps, learning_rate, "-")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Learning rate")

    ax2.plot(training_steps, loss_data, "-")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss")

    epochs = [x for x in range(1, num_epochs + 1, validate_freq)]

    ax3.plot(epochs, validation_score, "-")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Validation Score (Mean ROUGE-L F-measure)")

    plt.suptitle("Ketchup Model Training Metrics")
    plt.tight_layout()

    fig.delaxes(ax4)

    if save:
        plt.savefig("transformer_training_metrics.png")

    plt.show()

