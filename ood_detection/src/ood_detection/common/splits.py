"""OOD tier definitions, matching documentation/ood_split.md and the missions
baked into configs/grandtour*.yaml. Kept explicit here (rather than only reading
the config's flat NEAR_OOD_MISSIONS/FAR_OOD_MISSIONS lists) so tier-level AUROC
(near_forest vs near_gravel, far_snow vs far_arc) is always available, with an
assertion that catches drift if a config's split is ever edited."""

TIER_MISSIONS = {
    "near_forest": ["KAEB-1", "ROOT-1"],
    "near_gravel": ["EIG-1", "EIG-2", "PIL-2"],
    "far_snow": ["SNOW-1", "SNOW-2", "SNOW-3"],
    "far_arc": ["ARC-1", "ARC-2", "ARC-3", "ARC-4"],
}


def all_tiers(configs):
    near = set(configs.DATASET.NEAR_OOD_MISSIONS)
    far = set(configs.DATASET.FAR_OOD_MISSIONS)
    expected_near = set(TIER_MISSIONS["near_forest"]) | set(TIER_MISSIONS["near_gravel"])
    expected_far = set(TIER_MISSIONS["far_snow"]) | set(TIER_MISSIONS["far_arc"])
    assert near == expected_near, f"config NEAR_OOD_MISSIONS {near} != expected {expected_near}"
    assert far == expected_far, f"config FAR_OOD_MISSIONS {far} != expected {expected_far}"
    tiers = {"id_train": list(configs.DATASET.TRAIN_MISSIONS), "id_heldout": list(configs.DATASET.VALID_MISSIONS)}
    tiers.update(TIER_MISSIONS)
    return tiers
