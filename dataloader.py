from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


class EtymologyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers=2, pin_memory=True):

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
        )


class EtymologyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def df_to_array(df: pd.DataFrame):
    unique_related_lang = df["related_lang"].unique().tolist()
    print(f"Num related langs: {len(unique_related_lang)}")
    lang_to_i = {lang: i for i, lang in enumerate(unique_related_lang)}
    i_to_lang = {i: lang for i, lang in enumerate(unique_related_lang)}

    charset = set("".join(df["term"].tolist()))
    print(f"Charset size: {len(charset)}")
    char_to_i = {char: i for i, char in enumerate(charset)}
    i_to_char = {i: char for i, char in enumerate(charset)}

    input_vecs = [np.array([char_to_i[letter] for letter in term]) for term in df["term"]]
    # pad inputs
    max_input_len = max(len(term) for term in input_vecs)
    input_vecs = [np.pad(vec, (0, max_input_len - len(vec)), 'constant', constant_values=len(charset)) for vec in input_vecs]
    input_vecs = np.stack(input_vecs, axis=0)

    vocab_size = len(charset)
    num_output_classes = len(unique_related_lang)

    target_vec = np.array([lang_to_i[lang] for lang in df["related_lang"]])

    print(f"Dataset size: {len(input_vecs)}")

    return input_vecs, target_vec, i_to_lang, i_to_char, vocab_size + 1, num_output_classes
