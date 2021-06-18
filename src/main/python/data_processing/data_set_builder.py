from typing import Callable

import numpy as np
import random

from src.main.python.data_processing.data_transformation import load_raw_data_set, clean_single_entry, \
    save_clean_data_set, clean_single_entry_small_hashtag, clean_single_entry_small_accounts

base_dir = "/home/thrasolt/git/home_project_students"
data_path = base_dir+"/data/"
small_path = base_dir+"/data/small/"


def get_test_data_set():
    test_raw_data_set = load_raw_data_set(data_path + "test_raw_data.pickle")
    new_data_set = [clean_single_entry(row) for row in test_raw_data_set.values()]
    test_clean_data_set = np.array(new_data_set)
    save_clean_data_set(data_path+"test_data.pickle", test_clean_data_set)


def get_complete_set():
    raw_data_set_path = data_path + "raw_data.pickle"
    data_set_path = data_path + "data.pickle"

    raw_data_set = load_raw_data_set(raw_data_set_path)

    new_data_set = [clean_single_entry(row) for row in raw_data_set.values()]
    random.shuffle(new_data_set)
    random.shuffle(new_data_set)
    new_data_set = np.array(new_data_set)

    save_clean_data_set(data_set_path, new_data_set)


def get_small_set(transformation: Callable, file_name: str):
    raw_data_set_path = data_path + "raw_data.pickle"
    data_set_path = small_path + file_name

    raw_data_set = load_raw_data_set(raw_data_set_path)

    new_data_set = [transformation(row) for row in raw_data_set.values()]
    random.shuffle(new_data_set)
    random.shuffle(new_data_set)
    new_data_set = np.array(new_data_set)

    save_clean_data_set(data_set_path, new_data_set)


def get_small_data_sets():
    file_hashtag = "hashtag_data.pickle"
    file_account = "account_data.pickle"

    get_small_set(transformation=clean_single_entry_small_hashtag, file_name=file_hashtag)
    get_small_set(transformation=clean_single_entry_small_accounts, file_name=file_account)


if __name__ == "__main__":
    get_complete_set()
    get_small_data_sets()
    get_test_data_set()
