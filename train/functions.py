from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from typing import *
from datetime import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from string import punctuation
from string import ascii_letters as als
from random import randint
from sklearn.manifold import TSNE


def get_random_str(n):
    return ''.join([als[randint(0, len(als) - 1)] for _ in range(n)])


def get_tokens_sliced_by_punctuation(tokens: List[str], min_len=3, max_len=10):
    punkt = {k: 1 for k in punctuation}
    corpus, sentence = [], []
    for token in tokens:
        if punkt.get(token) or not token.isalpha():
            if len(sentence) >= min_len:
                corpus.append(sentence)
                sentence = []
        else:
            sentence.append(token.lower())

        if len(sentence) == max_len:
            corpus.append(sentence)
            sentence = []
    return corpus


def get_sequences_from_tokens(token_list: List[str], token2idx: Dict) -> List:
    token_indices = [token2idx[token] for token in token_list]
    sequences = [token_indices[:i + 1] for i in range(1, len(token_indices))]
    return sequences


def get_sequences_from_tokens_window(token_list: List[str], token2idx: Dict, window_size=2) -> List:
    token_indices = [token2idx[token] for token in token_list]
    length, w = len(token_indices), window_size
    sequences = [token_indices[i:i + w] for i in range(length - (w - 1))]
    return sequences


def get_sequences_matrix(tokens_matrix: List[List[str]], token2idx: Dict):
    sequences = []
    for tokens in tokens_matrix:
        sequences += get_sequences_from_tokens(tokens, token2idx)
    return sequences


def get_sequences_matrix_window(tokens_matrix: List[List[str]], token2idx: Dict, window_size=2):
    sequences = []
    for tokens in tokens_matrix:
        sequences += get_sequences_from_tokens_window(tokens, token2idx, window_size)
    return sequences


def pad_sequence_list(sequence: Iterable, max_len: int, method: str, truncating: str, value=0):
    assert isinstance(sequence, Iterable)
    sequence = list(sequence)
    if len(sequence) > max_len:
        if not truncating:
            raise Exception("The Length of a sequence is longer than max_len")
        if method == 'pre':
            return sequence[len(sequence) - max_len:]
        elif method == 'post':
            return sequence[:max_len - len(sequence)]
    else:
        if method == 'pre':
            return [value for _ in range(max_len - len(sequence))] + sequence
        elif method == 'post':
            return sequence + [value for _ in range(max_len - len(sequence))]


def pad_sequence_nested_lists(nested_sequence, max_len, method='pre', truncating='pre'):
    return [pad_sequence_list(seq, max_len, method, truncating) for seq in nested_sequence]


def process_line_strip(line, lower=True, rm_punct=True):
    splitted = line.strip().split(' ')
    strip_verse = splitted[2:]
    joined = ' '.join(strip_verse)
    joined = joined.lower() if lower else joined
    return joined


def process_multiple_lines_strip(lines, lower=True):
    return [process_line_strip(line, lower) for line in lines]


def to_categorical_one(index, length) -> np.ndarray:
    onehot = np.zeros(shape=(length,))
    onehot[index] = 1


def to_categorical_iterable(classes: Iterable, num_classes: int):
    assert isinstance(classes, Iterable)
    nrows, ncols = len(classes), num_classes
    onehot = np.zeros(shape=(nrows, ncols))
    onehot[range(nrows), classes] = 1
    return onehot


def get_max_seq_len(tokens_matrix):
    mx = 0
    for tokens in tokens_matrix:
        mx = max(mx, len(tokens))
    return mx


def get_uniques_from_nested_lists(nested_lists: List[List]) -> List:
    uniques = {}
    for one_line in nested_lists:
        for item in one_line:
            if not uniques.get(item):
                uniques[item] = 1
    return list(uniques.keys())


def get_item2idx(items, unique=False, start_from_one=False) -> Tuple[Dict, Dict]:
    item2idx, idx2item = dict(), dict()
    items_unique = items if unique else set(items)
    for idx, item in enumerate(items_unique):
        i = idx + 1 if start_from_one else idx
        item2idx[item] = i
        idx2item[i] = item
    if start_from_one:
        item2idx[0] = 0
        item2idx["<pad>"] = 0
        idx2item[0] = '<pad>'
    return item2idx, idx2item


def array_index_to_wv_padding(arr, wv, idx2token, dim_restrict=None):
    vectors, used_tokens = [], []
    sample = wv.get_vector('hello')[:dim_restrict]
    try:
        for elem in arr:
            if elem == 0:
                vectors.append(np.zeros_like(sample))
            else:
                used_tokens.append(idx2token[elem])
                vectors.append(wv.get_vector(idx2token[elem])[:dim_restrict])
        return np.array(vectors), used_tokens
    except Exception as e:
        print(str(e))
        return 'No vector', None


def array_index_to_wv_no_padding(arr, wv, idx2token, dim_restrict=None):
    cupy_type = "<class 'cupy.core.core.ndarray'>"
    is_cupy = True if str(type(arr)) == cupy_type else False
    vectors, used_tokens = [], []
    try:
        for elem in arr:
            elem = elem.item() if is_cupy else elem
            used_tokens.append(idx2token[elem])
            vectors.append(wv.get_vector(idx2token[elem])[:dim_restrict])
        return np.array(vectors), used_tokens
    except Exception as e:
        print(str(e))
        return 'No vector', None


def tsne_plot(labels, vectors, filename, perplexity=10, figsize=(8, 8), cmap='nipy_spectral', dpi=300):
    tsne_model = TSNE(perplexity=perplexity, n_components=2,
                      metric='cosine',
                      init='pca', n_iter=5000, random_state=22)
    new_values = tsne_model.fit_transform(vectors)

    x, y = [], []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.rcParams["font.family"] = 'D2Coding'
    plt.clf()
    plt.figure(figsize=figsize)
    plt.title(filename)
    plt.scatter(x, y, cmap=cmap, alpha=0.5)

    for i in range(len(x)):
        #         plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    timestamp = dt.today().strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig("results/{}_perp{}.png".format(filename, perplexity), dpi=dpi)





