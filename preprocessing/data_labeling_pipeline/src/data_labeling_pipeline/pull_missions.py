"""Download missions metadata from the Grand Tour website"""

import argparse
import json
import urllib.request
from pathlib import Path

MISSIONS_JSON_URL = "https://grand-tour.leggedrobotics.com/data/missions.json"
MISSIONS_DIR = Path(__file__).parent


def short_filter(data: dict) -> dict:
    include = ["name", "short_name", "labels", "split", "notes", "tags"]
    exclude_tags = ["Robot Name", "Deployment Day", "Google Drive Files", "GNSS"]

    x = {k: v for k, v in data.items() if k in include}
    x["tags"] = [t for t in x["tags"] if t["type"] not in exclude_tags]
    return x


def replace_umlauts(text: str) -> str:
    """Replace German umlauts in the text with their ASCII equivalents."""
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "AE",
        "Ö": "OE",
        "Ü": "UE",
    }
    for umlaut, replacement in replacements.items():
        text = text.replace(umlaut, replacement)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()

    output_name = "missions_short.json" if args.short else "missions.json"
    output_path = MISSIONS_DIR / output_name

    print("Fetching missions from Grand Tour website...")
    with urllib.request.urlopen(MISSIONS_JSON_URL) as resp:
        cleaned = replace_umlauts(resp.read().decode("utf-8"))
        data = json.loads(cleaned)

    missions = []
    for i, m in enumerate(data):
        entry = {k: v for k, v in m.items()}
        entry["id"] = i + 1
        if args.short:
            entry = short_filter(entry)
        missions.append(entry)

    output_path.write_text(json.dumps(missions, indent=4))
    print(f"Saved {len(missions)} missions to {output_path}")


if __name__ == "__main__":
    main()
