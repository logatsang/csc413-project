import argparse
from math import ceil
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm, trange

from transformer_model import Transformer
from dataloader import EtymologyDataLoader, EtymologyDataset, df_to_array

from model.cnn import EtymologyCNN
from model.rnn import EtymologyRNN
from model.transformer_encoder import EtymologyTransformerEncoder

DATA = r"data/etymology_top10.csv"
SEED = 3939

def build_cnn(vocab_size, num_classes, padding_idx, args):
    return EtymologyCNN(
        vocab_size=vocab_size,
        embedding_size=args.embedding_size,
        num_classes=num_classes,
        conv_filter_count=args.conv_filter_count,
        padding_idx=padding_idx,
        conv_layers=tuple(
            (args.conv_kernel_size, 1, 1, 2, 2, 0) for _ in range(args.num_layers)
        )
    )

def build_rnn(vocab_size, num_classes, padding_idx, args):
    return EtymologyRNN(
        vocab_size=vocab_size,
        embedding_size=args.embedding_size,
        hidden_size=args.embedding_size,
        num_classes=num_classes,
        num_layers=args.num_layers,
        padding_idx=padding_idx,
        rnn_type=args.rnn_type,
    )

def build_transformer(vocab_size, num_classes, padding_idx, args):
    return EtymologyTransformerEncoder(
        vocab_size=vocab_size,
        num_classes=num_classes,
        num_layers=args.num_layers,
        embedding_size=args.embedding_size,
        nhead=args.nhead,
        padding_idx=padding_idx,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default=DATA, help="Path to data CSV")
    parser.add_argument("-o", "--output", type=Path, help="Path to save images")
    parser.add_argument("--eval_every", type=int, default=5, help="Number of epochs to eval at")
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0, help="Initial learning rate")
    subparsers = parser.add_subparsers(required=True)

    parser_rnn = subparsers.add_parser("rnn", help="Train a RNN model")
    parser_rnn.add_argument("--rnn_type", choices=["rnn", "gru", "lstm"], default="gru")
    parser_rnn.add_argument("--num_layers", type=int, default=6)
    parser_rnn.set_defaults(factory=build_rnn)

    parser_cnn = subparsers.add_parser("cnn", help="Train a CNN model")
    parser_cnn.add_argument("--num_layers", type=int, default=1)
    parser_cnn.add_argument("--conv_filter_count", type=int, default=8)
    parser_cnn.add_argument("--conv_kernel_size", type=int, default=3)
    parser_cnn.set_defaults(factory=build_cnn)

    parser_trans = subparsers.add_parser("transformer", help="Train a transformer encoder model")
    parser_trans.add_argument("--num_layers", type=int, default=6)
    parser_trans.add_argument("--nhead", type=int, default=8)
    parser_trans.set_defaults(factory=build_transformer)

    args = parser.parse_args()

    if gpu := torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")

    df = pd.read_csv(args.input)
    input_vec, target_vec, i_to_lang, i_to_char, vocab_size, num_output_classes = df_to_array(df)
    padding_idx = vocab_size - 1

    X_train, X_test, y_train, y_test = train_test_split(input_vec, target_vec, test_size=0.4, stratify=target_vec, random_state=SEED)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=SEED)

    # Hyperparameters
    word_embedding_size = args.embedding_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.lr

    # model = EtymologyCNN(
    #     vocab_size=vocab_size,
    #     embedding_size=word_embedding_size,
    #     num_classes=num_output_classes,
    #     conv_layers=(
    #         (3, 1, 1, 2, 2, 0),
    #     ),
    #     conv_filter_count=16,
    #     padding_idx=padding_idx,
    # )

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

    model = args.factory(vocab_size, num_output_classes, padding_idx, args)

    print(vars(args))

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

    validate_freq = args.eval_every

    # scaler = GradScaler()

    # DATA
    learning_rate = []
    validation_score = []
    validation_roc = []
    validation_f1 = []
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
                all_logits = []
                all_labels = []
                with tqdm(total=len(val_dl), desc=f'Validating Epoch {epoch}', leave=False, unit='batch',
                          position=1, dynamic_ncols=True) as it:
                    for i, (X, y) in enumerate(val_dl):
                        X = X.to(device)
                        y = y.to(device)
                        all_labels.append(y.detach())

                        with torch.no_grad():
                            logits = model.forward(X)
                            all_logits.append(logits.detach())
                            output = torch.argmax(logits, dim=-1)
                            accurate += sum(output == y)

                            it.update()

                accuracy = accurate / len(y_val)
                tqdm.write(f"Validation accuracy after epoch {epoch}: {accuracy:.4%}")
                validation_score.append(accuracy.cpu())

                all_labels = torch.cat(all_labels, dim=0)
                all_logits = torch.cat(all_logits, dim=0)
                validation_roc.append(roc_auc_score(
                    all_labels.cpu(),
                    all_logits.softmax(dim=-1).cpu(),
                    multi_class="ovo",
                ))
                tqdm.write(f"Validation ROC AUC after epoch {epoch}: {validation_roc[-1]:.4}")
                validation_f1.append(f1_score(
                    all_labels.cpu(),
                    all_logits.argmax(dim=-1).cpu(),
                    average="macro",
                ))
                tqdm.write(f"Validation F1-score after epoch {epoch}: {validation_f1[-1]:.4}")

            pbar.bar_format = barfmt % epoch

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.8, 9.6))
    save = args.output is not None

    training_steps = [x for x in range(1, num_training_steps + 1)]

    ax1.plot(training_steps, loss_data, "-")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Training Loss")

    epochs = [x for x in range(1, num_epochs + 1, validate_freq)]

    ax2.plot(epochs, validation_score, "-")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")

    ax3.plot(epochs, validation_f1, "-")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Validation F1-score")

    ax4.plot(epochs, validation_roc, "-")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Validation ROC AUC")

    plt.suptitle("Model Training Metrics")
    plt.tight_layout()

    if save:
        args.output.mkdir(exist_ok=True)
        c = 0
        while True:
            outpath = args.output / f"img{c}.png"
            if not outpath.exists():
                break
            else:
                c += 1
        plt.savefig(outpath)
        print(f"Saved image to {str(outpath)}")
    else:
        plt.show()

