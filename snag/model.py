import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, JumpingKnowledge, global_mean_pool

from .search_space import act_map


class GATv2GraphNet(nn.Module):
    def __init__(self, actions, num_feat, num_label, args):
        super().__init__()
        self.args = args
        self.num_feat = num_feat
        self.num_label = num_label
        self.layer_nums = self.args.layers_of_child_model
        self.use_skip = []
        self.jk_mode = "none"
        self.layers = nn.ModuleList()
        self.acts = []
        self.dropouts = []
        self.layer_dims = []
        self.jk_func = None
        self.final_lin = None
        self.regressor = None

        self._build(actions)

    def _get_shared_layer(self, key, in_dim, out_dim, heads, concat, dropout):
        layer = GATv2Conv(
            in_dim,
            out_dim,
            heads=heads,
            concat=concat,
            dropout=dropout,
        )
        if self.args.shared_params and self.args.update_shared:
            self.args.shared_parms_dict[key] = layer
        return layer

    def _resolve_layer(self, key, in_dim, out_dim, heads, concat, dropout):
        if not self.args.shared_params:
            return GATv2Conv(in_dim, out_dim, heads=heads, concat=concat, dropout=dropout)

        if key in self.args.shared_parms_dict:
            if self.args.update_shared:
                return self.args.shared_parms_dict[key]
            return copy.deepcopy(self.args.shared_parms_dict[key])

        return self._get_shared_layer(key, in_dim, out_dim, heads, concat, dropout)

    def _build(self, actions):
        if self.layer_nums < 1:
            raise ValueError("layers_of_child_model must be >= 1")

        heads = actions[0]
        hidden_dim = actions[1]
        dropout = actions[2]
        act_list = actions[3:3 + self.layer_nums]
        skip_offset = 3 + self.layer_nums
        skip_list = actions[skip_offset:skip_offset + max(0, self.layer_nums - 1)]
        self.use_skip = skip_list + [1]
        self.jk_mode = actions[-1]

        in_dim = self.num_feat
        out_dim = hidden_dim

        for i in range(self.layer_nums):
            act = act_list[i]
            concat = True
            key = f"{i}_{in_dim}_{out_dim}_{heads}_{concat}_gatv2"
            layer = self._resolve_layer(key, in_dim, out_dim, heads, concat, dropout)
            self.layers.append(layer)
            self.acts.append(act_map(act))
            self.dropouts.append(dropout)
            layer_out_dim = out_dim * heads if concat else out_dim
            self.layer_dims.append(layer_out_dim)
            in_dim = layer_out_dim

        if self.jk_mode != "none":
            if self.jk_mode == "concat":
                self.jk_func = JumpingKnowledge(mode="cat")
                jk_out_dim = sum(
                    dim for dim, use in zip(self.layer_dims, self.use_skip) if use
                )
            elif self.jk_mode == "maxpool":
                jk_out_dim = self._check_equal_dims()
                self.jk_func = JumpingKnowledge(mode="max")
            elif self.jk_mode == "lstm":
                jk_out_dim = self._check_equal_dims()
                self.jk_func = JumpingKnowledge(
                    mode="lstm",
                    channels=jk_out_dim,
                    num_layers=sum(self.use_skip),
                )
            else:
                raise ValueError(f"Unknown jk_mode: {self.jk_mode}")
        else:
            jk_out_dim = self.layer_dims[-1]

        self.final_lin = nn.Linear(jk_out_dim, jk_out_dim)
        self.regressor = nn.Linear(jk_out_dim, self.num_label)
        self.dropout = dropout

    def _check_equal_dims(self):
        selected_dims = [
            dim for dim, use in zip(self.layer_dims, self.use_skip) if use
        ]
        if not selected_dims:
            return self.layer_dims[-1]
        if len(set(selected_dims)) != 1:
            raise ValueError("JK mode requires equal layer dims; use concat or fix hidden/head")
        return selected_dims[0]

    def forward(self, x, edge_index, batch):
        outputs = []
        for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
            x = F.dropout(x, p=self.dropouts[i], training=self.training)
            x = layer(x, edge_index)
            x = act(x)
            if self.use_skip[i]:
                outputs.append(x)

        if self.jk_mode != "none":
            if not outputs:
                outputs = [x]
            x = self.jk_func(outputs)

        graph_emb = global_mean_pool(x, batch)
        graph_emb = self.final_lin(graph_emb)
        graph_emb = F.dropout(graph_emb, p=self.dropout, training=self.training)
        out = self.regressor(graph_emb).view(-1)
        return out
