"""Inspect ROS1 bags and write ros_topics.md."""

import fnmatch
import pathlib
import typing
from collections import defaultdict

from shared_utils.paths import ROS1_ROOT
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

MISSION_PATH = ROS1_ROOT / "ETH-1"
OUTPUT_PATH = pathlib.Path(__file__).parent.parent / "ros_topics.md"
# Bag name filters: globs if they contain *?[, else substrings (e.g. "alphasense")
PATTERNS = ["*"]

typestore = get_typestore(Stores.ROS1_NOETIC)

if "tf2_msgs/msg/TFMessage" not in typestore.types:
    tf2_msg_def = "geometry_msgs/TransformStamped[] transforms"
    typestore.register(get_types_from_msg(tf2_msg_def, "tf2_msgs/msg/TFMessage"))

# Not expanded in type definitions
ATOMIC_TYPES = {
    "std_msgs/Header",
    "geometry_msgs/Vector3",
    "geometry_msgs/Point",
    "geometry_msgs/Quaternion",
    "geometry_msgs/Pose",
    "geometry_msgs/Twist",
}


def clean_msgtype(msgtype: str) -> str:
    if not msgtype:
        return "?"
    parts = msgtype.split("/")
    if "msg" in parts:
        parts.remove("msg")
    if "srv" in parts:
        parts.remove("srv")
    return "/".join(parts)


def get_msgtype_for_class(cls):
    """Find the ROS msgtype string for a given class."""
    if not cls:
        return None

    for name, t in typestore.types.items():
        if t == cls:
            return name

    if hasattr(cls, "__name__"):
        name = cls.__name__
        if "__" in name:
            # geometry_msgs__msg__Pose -> geometry_msgs/msg/Pose
            return name.replace("__", "/")
    return str(cls)


def type_name(t) -> str:
    if t is None:
        return "?"
    if t is type(None):
        return "None"

    if typing.get_origin(t) is typing.Annotated:
        return type_name(typing.get_args(t)[0])

    origin = getattr(t, "__origin__", None)
    if origin is list:
        args = getattr(t, "__args__", ())
        inner = type_name(args[0]) if args else "?"
        return f"list[{inner}]"

    if hasattr(t, "__name__"):
        mod = getattr(t, "__module__", "")
        if mod == "numpy":
            return t.__name__

        mtype = get_msgtype_for_class(t)
        if mtype:
            return clean_msgtype(mtype)
        return t.__name__

    return str(t)


def is_ros_msg(t) -> bool:
    if not isinstance(t, type):
        return False
    return hasattr(t, "__annotations__")


def unwrap_type(t):
    """Unwrap Annotated / list wrappers to get the core element type."""
    if typing.get_origin(t) is typing.Annotated:
        return unwrap_type(typing.get_args(t)[0])
    origin = getattr(t, "__origin__", None)
    if origin is list:
        args = getattr(t, "__args__", ())
        if args:
            return unwrap_type(args[0])
    return t


def is_complex_type(cls):
    """ROS msg type that is not in ATOMIC_TYPES."""
    if not is_ros_msg(cls):
        return False

    mtype = clean_msgtype(get_msgtype_for_class(cls))
    if mtype in ATOMIC_TYPES:
        return False

    return True


def collect_fields(cls, seen_complex_classes: set):
    """Field lines of a class; complex subtypes are added to the set."""
    if not is_ros_msg(cls):
        return []

    lines = []
    annotations = getattr(cls, "__annotations__", {})

    for field_name, hint in annotations.items():
        if field_name.startswith("__"):
            continue
        if field_name == field_name.upper() and "_" in field_name:
            continue

        core = unwrap_type(hint)
        tname = type_name(hint)
        lines.append(f"- `{field_name}` ({tname})")

        if is_complex_type(core):
            seen_complex_classes.add(core)

    return lines


def find_class(msgtype: str):
    if msgtype in typestore.types:
        return typestore.types[msgtype]

    target = msgtype.replace("/msg/", "/").replace("/srv/", "/")
    for name, cls in typestore.types.items():
        if name.replace("/msg/", "/").replace("/srv/", "/") == target:
            return cls
    return None


def bag_matches_patterns(bag_path: pathlib.Path) -> bool:
    stem = bag_path.stem.lower()
    for p in PATTERNS:
        if not p:
            continue
        p_low = p.lower()
        if any(ch in p_low for ch in "*?["):
            if fnmatch.fnmatch(stem, p_low):
                return True
        elif p_low in stem:
            return True
    return False


bags = sorted(f for f in MISSION_PATH.glob("*.bag") if bag_matches_patterns(f))

if not bags:
    print(f"No matching bags found in {MISSION_PATH}")
    print(f"Expected patterns: {PATTERNS}")
    print("Run download_wayfaster_data.py first.")
    raise SystemExit(1)

all_complex_classes = set()
bag_content_lines = []

for bag_path in bags:
    stem = bag_path.stem
    _, _, rest = stem.partition("_")
    bag_label = f"*_{rest}.bag"
    print(f"Inspecting {bag_path.name}...")

    topic_infos = []
    tf_infos = []

    with Reader(bag_path) as reader:
        by_topic = defaultdict(list)
        for conn in reader.connections:
            by_topic[conn.topic].append(conn)

        for topic, conns in sorted(by_topic.items()):
            msgtype = conns[0].msgtype

            if "TFMessage" in msgtype:
                pairs = set()
                for conn, timestamp, rawdata in reader.messages(connections=conns):
                    try:
                        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                        for tr in msg.transforms:
                            pairs.add((
                                tr.header.frame_id.strip("/"),
                                tr.child_frame_id.strip("/"),
                            ))
                    except Exception:
                        pass
                tf_infos.append((topic, msgtype, sorted(pairs)))
            else:
                frame_str = "N/A"
                for conn, timestamp, rawdata in reader.messages(connections=conns):
                    try:
                        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                        frame_id = getattr(
                            getattr(msg, "header", None), "frame_id", None
                        )
                        child_frame_id = getattr(msg, "child_frame_id", None)
                        if frame_id:
                            frame_str = frame_id
                            if child_frame_id:
                                frame_str += f" → {child_frame_id}"
                    except Exception:
                        frame_str = "(unregistered type)"
                    break

                cls = find_class(msgtype)
                field_lines = (
                    collect_fields(cls, all_complex_classes)
                    if cls
                    else ["- *(type not registered)*"]
                )
                topic_infos.append((topic, msgtype, frame_str, field_lines))

    bag_content_lines.append("---")
    bag_content_lines.append("")
    bag_content_lines.append(f"## `{bag_label}`")
    bag_content_lines.append("")

    for topic, msgtype, frame_str, field_lines in topic_infos:
        msgtype_short = clean_msgtype(msgtype)
        bag_content_lines.append(f"### `{topic}`")
        bag_content_lines.append(f"- **Type:** `{msgtype_short}`")
        bag_content_lines.append(f"- **Frame:** `{frame_str}`")
        bag_content_lines.append("- **Fields:**")
        bag_content_lines.extend(field_lines)
        bag_content_lines.append("")

    for topic, msgtype, pairs in tf_infos:
        msgtype_short = clean_msgtype(msgtype)
        bag_content_lines.append(f"### `{topic}`")
        bag_content_lines.append(f"- **Type:** `{msgtype_short}`")
        bag_content_lines.append("- **Transform pairs (parent → child):**")
        for parent, child in pairs:
            bag_content_lines.append(f"  - `{parent}` → `{child}`")
        bag_content_lines.append("")

# Expand nested subtypes until no new ones appear
expanded_types = {}
while True:
    new_classes = all_complex_classes - set(expanded_types.keys())
    if not new_classes:
        break
    for cls in sorted(
        new_classes, key=lambda c: get_msgtype_for_class(c) or c.__name__
    ):
        expanded_types[cls] = collect_fields(cls, all_complex_classes)

final_lines = [
    "# GrandTour Dataset — ROS1 Bag Topics",
    "",
    "Each mission directory contains a set of bag files. "
    "All bags are prefixed with the mission timestamp (e.g. `2024-10-01-11-29-55_`).",
    "",
]
final_lines.extend(bag_content_lines)

if expanded_types:
    final_lines.append("---")
    final_lines.append("")
    final_lines.append("## Type Definitions")
    final_lines.append("")
    for cls, flines in expanded_types.items():
        mtype = clean_msgtype(get_msgtype_for_class(cls))
        final_lines.append(f"### `{mtype}`")
        final_lines.extend(flines)
        final_lines.append("")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(final_lines))
print(f"Written to {OUTPUT_PATH}")
