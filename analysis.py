from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from transformer_runner import df_to_array

if __name__ == "__main__":
    df = pd.read_csv("data/etymology_proc.csv")
    input_vec, target_vec, i_to_lang, i_to_char, vocab_size, _ = df_to_array(df)

    count_lang = defaultdict(int)
    for lang in target_vec:
        count_lang[lang] += 1

    ranked_counts = np.log(np.array(sorted((val for val in count_lang.values()), reverse=True)))
    ranks = np.log(np.arange(1, len(ranked_counts) + 1))

    fig, ax = plt.subplots()

    ax.scatter(ranks, ranked_counts, s=3)
    ax.axline((8, 0), slope=-1, color='red', alpha=0.5)

    ax.set_ylabel("Frequency (log)")
    ax.set_xlabel("Rank (log)")

    plt.title("Zipf's Law on languages related to English")

    plt.show()

    # replacement = {i_to_lang[i]: "other" for i in i_to_lang.keys() if count_lang[i] < 2**4}
    # print(replacement)
    # print(len(replacement))
    # rdf = df.replace(replacement)
    # rdf.to_csv('data/etymology_proc_pooled.csv', index=False)
