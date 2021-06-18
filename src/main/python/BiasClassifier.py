import csv
import pickle
from typing import List

import torch

from Classifier.Model import TwoLaneModel
from src.main.python.data_retrieval.file_reader import test_path, read_plds

base_dir = "/home/thrasolt/git/home_project_students"
model_path = base_dir + "/models/" + "final_model.pytorch"

in_dim = 906
out_dim = 2

inner_layers = [64, 32]

text_dim = 900
text_layers = [256, 64]
emotions_dim = 6
emotions_layers = [16, 16]
combination_dim = 16
combination_layers = [32, 16]


class BiasClassifier:
    def __init__(self):
        self.model = TwoLaneModel(
            text_dim, text_layers,
            emotions_dim,
            emotions_layers,
            combination_dim,
            combination_layers,
            out_dim)
        # self.model = SingleLaneModel(in_dim=in_dim, out_dim=out_dim, inner_layers=inner_layers)
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        except Exception as exception:
            print("Could not load trained model", exception)

    @torch.no_grad()
    def classify_test_data(self, pld_list: List[str], out_path_left: str, out_path_right):

        test_data_set_path = base_dir + "/data/" + "test_data.pickle"

        with open(test_data_set_path, 'rb') as file_handle:
            data_and_labels = pickle.load(file_handle)

        data = data_and_labels[:, :-1]
        output = self.model(torch.tensor(data))
        results = output.detach().numpy()

        right_scores = []
        left_scores = []

        for (pld, result) in zip(pld_list, results):
            left_score, right_score = result[0].item(), result[1].item()
            if left_score >= right_score:
                left_scores.append([pld, left_score])
            elif right_score >= left_score:
                right_scores.append([pld, right_score])

        with open(out_path_left, "w") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerows(left_scores)

        with open(out_path_right, "w") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerows(right_scores)


if __name__ == "__main__":
    test_plds = [pld for (pld, label) in read_plds(test_path)]
    test_right_path = base_dir + "/output_data/right.csv"
    test_left_path = base_dir + "/output_data/left.csv"

    bias_classifier = BiasClassifier()
    bias_classifier.classify_test_data(test_plds, test_left_path, test_right_path)
