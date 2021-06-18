from typing import List

import torch
import torch.nn as nn

base_dir = "/home/thrasolt/git/home_project_students"


class TwoLaneModel(nn.Module):
    def __init__(self,
                 text_dim: int,
                 text_layers: List[int],
                 emotions_dim: int,
                 emotions_layers: List[int],
                 combination_dim: int,
                 combination_layers: List[int],
                 out_dim: int):
        super().__init__()

        self.text_dim = text_dim
        self.emotions_dim = emotions_dim
        self.combination_dim = combination_dim
        self.out_dim = out_dim

        self.text_lane = SingleLaneModel(text_dim, combination_dim, text_layers)
        self.emo_lane = SingleLaneModel(emotions_dim, combination_dim, emotions_layers)
        self.combination_lane = SingleLaneModel(2*combination_dim, out_dim, combination_layers)

        self.relu = nn.ReLU()

    def forward(self, input_data: torch.Tensor):
        input_data = input_data.float()
        text = input_data[:, :self.text_dim]
        emotions = input_data[:, self.text_dim:]

        text_result = self.relu(self.text_lane(text))
        emotions_result = self.relu(self.emo_lane(emotions))

        combination = torch.cat([text_result, emotions_result], 1)
        output = self.combination_lane(combination)

        return output


class SingleLaneModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, inner_layers: List[int]):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inner_layers = inner_layers

        hidden_layers = list()
        for inn, out in zip(inner_layers[:-1], inner_layers[1:]):
            hidden_layers.append(nn.Linear(inn, out))
            hidden_layers.append(nn.ReLU())

        self.batch_norm = nn.BatchNorm1d(in_dim)
        self.layer_in = nn.utils.weight_norm(nn.Linear(in_dim, inner_layers[0]))
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.layer_out = nn.utils.weight_norm(nn.Linear(inner_layers[-1], out_dim))

        self.relu = nn.ReLU()

    def forward(self, input_data: torch.Tensor):
        input_data = input_data.float()

        intermediate = self.batch_norm(input_data)
        intermediate = self.relu(self.layer_in(intermediate))
        intermediate = self.hidden_layers(intermediate)
        output = self.layer_out(intermediate)

        return output


