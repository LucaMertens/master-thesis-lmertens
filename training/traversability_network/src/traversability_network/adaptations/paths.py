"""Per-mission media root resolution for the resized-image and depth caches."""
import os


def resolve_media_roots(curr_dir, resized_root, depth_cache_root):
    """(img_root, depth_root) for a mission: the cache dir if it exists there, else curr_dir."""
    img_root = curr_dir
    if resized_root:
        candidate = os.path.join(resized_root, os.path.basename(curr_dir))
        if os.path.isdir(candidate):
            img_root = candidate
    depth_root = curr_dir
    if depth_cache_root:
        depth_root = os.path.join(depth_cache_root, os.path.basename(curr_dir))
    return img_root, depth_root
