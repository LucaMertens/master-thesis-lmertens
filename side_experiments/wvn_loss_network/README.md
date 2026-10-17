# wvn_loss_network

WVN-style OOD loss for the traversability network. A Mahalanobis gate on frozen DINO features scores the "OOD-ness" of image regions, and an extra loss pulls the predicted mu at those BEV cells towards zero.


## Usage

```bash
uv run python side_experiments/wvn_loss_network/gate_check.py
uv run python side_experiments/wvn_loss_network/train_wvn.py --cfg_file side_experiments/wvn_loss_network/wvn_dino.yaml
```
