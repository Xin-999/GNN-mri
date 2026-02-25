import sys
from pathlib import Path

"""
Sample usage (from repo root):
    python training/snag/train_snag_gatv2.py --device cuda --train_epochs 20
    python training/snag/train_snag_gatv2.py --device cpu --fold_name graphs_outer0
    python training/snag/train_snag_gatv2.py --heads 1,2,4 --hidden_dims 32,64 --dropouts 0.0,0.2
    python training/snag/train_snag_gatv2.py --device cuda --train_epochs 20 --epochs 40 --patience 8 \
        --layers_of_child_model 3 --derive_num_sample 20 --controller_max_step 10 --heads 2,4 \
        --hidden_dims 64,128 --dropouts 0.2,0.4 --activations relu,elu --jk_modes none,concat

Key args (more detail):
  --derive_num_sample:
      How many architectures are sampled in the final "derive" phase.
      The best one (by subject-level validation Pearson r) is chosen.
      Higher = better chance to find a good model, but slower. Default 10.
  --controller_max_step:
      How many controller update steps happen per controller epoch.
      Each step samples 1 architecture (controller batch size = 1) and updates
      the policy. Higher = more exploration per epoch, more compute. Default 10.
  --jk_modes:
      Jumping Knowledge strategy applied before graph pooling:
        none    = use last layer only
        concat  = concatenate selected layer outputs
        maxpool = elementwise max over selected outputs (dims must match)
        lstm    = LSTM over selected outputs (dims must match)
      The controller picks one from the list you provide. Default all above.
  --layers_of_child_model:
      Number of GATv2 message-passing layers in each sampled child model.
      Also controls how many per-layer choices the controller makes
      (activation + optional skip for each layer). Higher = deeper models,
      more compute/memory, and a larger search space. Default 3.
"""

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from snag.main import main


if __name__ == "__main__":
    main()
