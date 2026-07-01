

```bash
uv run --project training/traversability_network python -u -m traversability_network.train --cfg_file training/traversability_network/configs/grandtour_lite.yaml 2>&1 | tee train.log
```