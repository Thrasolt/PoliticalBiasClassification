from typing import List, Dict

import pickle

from src.main.python.data_retrieval.data_retrieval import retrieve_data_for_pld
from src.main.python.data_retrieval.file_reader import read_training_data, read_plds


def load_raw_data_set(path: str) -> Dict:
    with open(path, 'rb') as file_handle:
        return pickle.load(file_handle)


def save_raw_data_set(path: str, data: Dict):
    with open(path, 'wb') as file_handle:
        pickle.dump(data, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def collect_data(raw_data_set_path: str, pld_list):
    data_set = load_raw_data_set(raw_data_set_path)

    current_total_size = len(data_set)
    print("Data Size:", current_total_size)

    index = 1
    limit = 30
    batch_size = 3

    total_data_size = len(pld_list)

    for pld, label in pld_list:

        if pld in data_set:
            continue

        print(f"Total: {current_total_size + index} / {total_data_size}, current run: {index} / {limit} - {pld}")
        entry: List = retrieve_data_for_pld(pld) + [label]
        data_set[pld] = entry

        if index % batch_size == 0:
            save_raw_data_set(raw_data_set_path, data_set)
            print(f"saved at total index {current_total_size + index}")
        if index >= limit:
            break
        index += 1

    save_raw_data_set(raw_data_set_path, data_set)


if __name__ == "__main__":

    raw_data_set_path = "/home/thrasolt/git/home_project_students/data/" + "raw_data.pickle"
    training_plds = read_training_data()

    test_raw_data_set_path = "/home/thrasolt/git/home_project_students/data/" + "test_raw_data.pickle"
    test_plds = read_plds()
    for _ in range(30):
        try:
            collect_data(raw_data_set_path, training_plds)
            collect_data(test_raw_data_set_path, test_plds)
        except Exception as exception:
            print(exception)
