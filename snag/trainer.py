import time
import numpy as np
import torch

from .controller import SimpleNASController
from .manager import GATv2RegressionManager
from .search_space import GATv2SearchSpace
from .utils import tensor_utils as utils


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.cuda = args.cuda
        self.controller = None
        self.submodel_manager = None
        self.search_space = None
        self.action_list = None
        self.controller_optim = None
        self.history = []
        self._build_model()

    def _build_model(self):
        search_space_cls = GATv2SearchSpace()
        merged_space = search_space_cls.get_search_space()
        if getattr(self.args, "search_space", None):
            merged_space.update(self.args.search_space)
            search_space_cls = GATv2SearchSpace(search_space=merged_space)

        self.search_space = search_space_cls.get_search_space()
        self.action_list = search_space_cls.generate_action_list(self.args.layers_of_child_model)

        self.controller = SimpleNASController(
            self.args,
            action_list=self.action_list,
            search_space=self.search_space,
            cuda=self.args.cuda,
            controller_hid=self.args.controller_hid,
        )

        if not hasattr(self.args, "shared_parms_dict"):
            self.args.shared_parms_dict = {}
        if not hasattr(self.args, "update_shared"):
            self.args.update_shared = False

        self.submodel_manager = GATv2RegressionManager(self.args)
        if self.cuda:
            self.controller.cuda()

        optim_cls = torch.optim.Adam
        if self.args.controller_optim == "sgd":
            optim_cls = torch.optim.SGD
        self.controller_optim = optim_cls(
            self.controller.parameters(), lr=self.args.controller_lr
        )

    def train_shared(self, max_step=0):
        if max_step == 0 or not self.args.shared_params:
            return
        gnn_list = self.controller.sample(max_step)
        for gnn in gnn_list:
            self.submodel_manager.train(gnn, evaluate_test=False)

    def get_reward(self, gnn_list, entropies):
        if isinstance(gnn_list, dict):
            gnn_list = [gnn_list]
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()

        rewards = []
        for gnn in gnn_list:
            if self.args.shared_params:
                val_score, metrics = self.submodel_manager.evaluate(gnn)
            else:
                train_result = self.submodel_manager.train(gnn, evaluate_test=False)
                val_score = train_result[\"val_score\"]
                metrics = train_result[\"val_metrics\"]
            rewards.append(val_score)
            self.history.append({
                "actions": gnn,
                "val_score": float(val_score),
                "metrics": metrics,
            })

        rewards = np.array(rewards)
        if self.args.entropy_mode == "reward":
            rewards = rewards + self.args.entropy_coeff * entropies
        return rewards

    def train_controller(self):
        self.controller.train()
        baseline = None
        for _ in range(self.args.controller_max_step):
            gnn_list, log_probs, entropies = self.controller.sample(with_details=True)
            rewards = self.get_reward(gnn_list, entropies)

            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv = utils.get_variable(adv, self.cuda, requires_grad=False)
            loss = -(log_probs * adv).sum()

            self.controller_optim.zero_grad()
            loss.backward()
            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(
                    self.controller.parameters(), self.args.controller_grad_clip
                )
            self.controller_optim.step()

    def derive(self, sample_num=None):
        if sample_num is None:
            sample_num = self.args.derive_num_sample
        gnn_list, _, _ = self.controller.sample(sample_num, with_details=True)

        best_actions = None
        best_score = -float("inf")
        for gnn in gnn_list:
            val_score, _ = self.submodel_manager.evaluate(gnn)
            if val_score > best_score:
                best_score = val_score
                best_actions = gnn

        return best_actions, best_score

    def finetune(self, actions):
        self.args.shared_params = False
        self.args.update_shared = False
        result = self.submodel_manager.train(actions, evaluate_test=True)
        return result

    def train(self):
        self.args.shared_parms_dict = {}
        self.args.update_shared = True

        start_time = time.time()
        for _ in range(self.args.train_epochs):
            self.args.update_shared = True
            self.train_shared(max_step=self.args.shared_initial_step)

            self.args.update_shared = False
            self.train_controller()

            if self.args.time_budget > 0:
                elapsed = (time.time() - start_time) / 3600.0
                if elapsed >= self.args.time_budget:
                    break

        best_actions, best_score = self.derive()
        finetune_result = self.finetune(best_actions)

        return {
            "best_actions": best_actions,
            "best_val_score": float(best_score),
            "finetune": finetune_result,
            "history": self.history,
        }
