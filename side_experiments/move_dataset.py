import re
import shutil
import tarfile
from pathlib import Path


def move_dataset(cache, dataset_folder, allow_patterns=["*"]):
    """Move files downloaded from huggingface from cache to dataset folder, while allowing only certain patterns."""
    print(f"Start moving from {cache} to {dataset_folder} !")

    def convert_glob_patterns_to_regex(glob_patterns):
        regex_parts = []
        for pat in glob_patterns:
            # Escape regex special characters except for * and ?
            pat = re.escape(pat)
            # Convert escaped glob wildcards to regex equivalents
            pat = pat.replace(r"\*", ".*").replace(r"\?", ".")
            # Make sure it matches full paths
            regex_parts.append(f".*{pat}$")

        # Join with |
        combined = "|".join(regex_parts)
        return re.compile(combined)

    pattern = convert_glob_patterns_to_regex(allow_patterns)
    files = [f for f in Path(cache).rglob("*") if pattern.match(str(f))]
    tar_files = [f for f in files if f.suffix == ".tar"]

    for source_path in tar_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(source_path, "r") as tar:
                tar.extractall(path=dest_path.parent)
        except tarfile.ReadError as e:
            print(f"Error opening or extracting tar file '{source_path}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {source_path}: {e}")

    other_files = [f for f in files if not f.suffix == ".tar" and f.is_file()]
    for source_path in other_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

    print(f"Moved data from {cache} to {dataset_folder} !")
