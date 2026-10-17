"""Extraction-side view of shared_utils/tiers.json. ID tiers come from the checkpoint config and are checked against the file."""

from shared_utils.tiers import ID_TRAIN, tier_missions, time_ranges

HELDOUT_TIER = "id_heldout"


def tier_time_ranges(tier):
    """mission -> absolute-second spans to extract, empty for a whole tier."""
    return time_ranges(tier)


def all_tiers(configs):
    """tier -> missions to extract, ID first and then near to far."""
    tiers = tier_missions(configs.DATASET.TRAIN_MISSIONS)
    for tier, missions in ((ID_TRAIN, configs.DATASET.TRAIN_MISSIONS),
                           (HELDOUT_TIER, configs.DATASET.VALID_MISSIONS)):
        assert set(tiers.get(tier, [])) == set(missions), (
            f"config {tier} {sorted(missions)} != tiers.json {sorted(tiers.get(tier, []))}")
    return tiers


# OOD tiers only, for callers without a checkpoint config
TIER_MISSIONS = {t: ms for t, ms in tier_missions().items()
                 if t not in (ID_TRAIN, HELDOUT_TIER)}
