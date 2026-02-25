import torch


class GATv2SearchSpace(object):
    def __init__(self, search_space=None):
        if search_space:
            self.search_space = search_space
        else:
            self.search_space = {
                "heads": [1, 2, 4, 8],
                "hidden_dim": [32, 64, 128],
                "dropout": [0.0, 0.2, 0.5],
                "activation": ["relu", "elu"],
                "use_skip": [0, 1],
                "jk_mode": ["none", "concat", "maxpool", "lstm"],
            }

    def get_search_space(self):
        return self.search_space

    def generate_action_list(self, num_of_layers=2):
        action_list = ["heads", "hidden_dim", "dropout"]
        action_list += ["activation"] * num_of_layers
        if num_of_layers > 1:
            action_list += ["use_skip"] * (num_of_layers - 1)
        action_list += ["jk_mode"]
        return action_list


def act_map(act):
    if act == "linear":
        return lambda x: x
    if act == "elu":
        return torch.nn.functional.elu
    if act == "sigmoid":
        return torch.sigmoid
    if act == "tanh":
        return torch.tanh
    if act == "relu":
        return torch.nn.functional.relu
    if act == "relu6":
        return torch.nn.functional.relu6
    if act == "softplus":
        return torch.nn.functional.softplus
    if act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    raise ValueError(f"Unknown activation: {act}")
