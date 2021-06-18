import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch

from torchtext.vocab import FastText
from torchtext.data.utils import get_tokenizer
embedding = FastText('simple')
tokenizer = get_tokenizer('basic_english')

def load_raw_data_set(path: str) -> Dict:
    with open(path, 'rb') as file_handle:
        return pickle.load(file_handle)


def save_clean_data_set(path: str, data: np.ndarray):
    with open(path, 'wb') as file_handle:
        pickle.dump(data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def aggregate_emotions_mean(emotions: List[str]) -> (np.ndarray, np.ndarray, int):
    emotion_intensities = np.array(emotions).astype(np.float)
    return emotion_intensities.mean(), emotion_intensities.std(), len(emotion_intensities)

def words_to_numbers(words: List[str]):
    tokens = tokenizer(" ".join(words))
    value = torch.zeros(300)

    for token in tokens:
        value += embedding[token]

    return value


def clean_single_entry_small_hashtag(row: Tuple):
    hashtags = words_to_numbers(row[1]).numpy()
    pos_emo_mean, _, _ = aggregate_emotions_mean(row[2])
    neg_emo_mean, _, _ = aggregate_emotions_mean(row[3])
    label = row[6]

    return np.hstack(
        [hashtags,
         pos_emo_mean,
         neg_emo_mean,
         label
         ])


def clean_single_entry_small_accounts(row: Tuple):
    accounts = words_to_numbers(row[5]).numpy()
    pos_emo_mean, _, _ = aggregate_emotions_mean(row[2])
    neg_emo_mean, _, _ = aggregate_emotions_mean(row[3])
    label = row[6]

    return np.hstack(
        [accounts,
         pos_emo_mean,
         neg_emo_mean,
         label
         ])


def clean_single_entry(row: Tuple):
    hashtags = words_to_numbers(row[1]).numpy()
    entities = words_to_numbers(row[4]).numpy()
    accounts = words_to_numbers(row[5]).numpy()
    pos_emo_mean, pos_emo_std, pos_emo_count = aggregate_emotions_mean(row[2])
    neg_emo_mean, neg_emo_std, neg_emo_count = aggregate_emotions_mean(row[3])
    label = row[6]

    return np.hstack(
        [hashtags,
         entities,
         accounts,
         pos_emo_mean,
         pos_emo_std,
         pos_emo_count,
         neg_emo_mean,
         neg_emo_std,
         neg_emo_count,
         label
         ])


