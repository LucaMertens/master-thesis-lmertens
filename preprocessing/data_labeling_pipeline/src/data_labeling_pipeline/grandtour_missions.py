# This file is used to maintain a fixed mapping between mission properties.

missions = [
    {"id": 1, "short_name": "ETH-1", "timestamp": "2024-10-01-11-29-55"},
    {"id": 2, "short_name": "ETH-2", "timestamp": "2024-10-01-11-47-44"},
    {"id": 3, "short_name": "ETH-3", "timestamp": "2024-10-01-12-00-49"},
    {"id": 4, "short_name": "SPX-1", "timestamp": "2024-11-02-17-10-25"},
    {"id": 5, "short_name": "SPX-2", "timestamp": "2024-11-02-17-18-32"},
    {"id": 6, "short_name": "SPX-3", "timestamp": "2024-11-02-17-43-10"},
    {"id": 7, "short_name": "ICE-1", "timestamp": "2024-11-02-21-12-51"},
    {"id": 8, "short_name": "SNOW-1", "timestamp": "2024-11-03-07-52-45"},
    {"id": 9, "short_name": "SNOW-2", "timestamp": "2024-11-03-07-57-34"},
    {"id": 10, "short_name": "SNOW-3", "timestamp": "2024-11-03-08-17-23"},
    {"id": 11, "short_name": "EIG-1", "timestamp": "2024-11-03-13-51-43"},
    {"id": 12, "short_name": "EIG-2", "timestamp": "2024-11-03-13-59-54"},
    {"id": 13, "short_name": "GRI-1", "timestamp": "2024-11-04-10-57-34"},
    {"id": 14, "short_name": "CYN-1", "timestamp": "2024-11-04-12-55-59"},
    {"id": 15, "short_name": "CYN-2", "timestamp": "2024-11-04-13-07-13"},
    {"id": 16, "short_name": "HIL-1", "timestamp": "2024-11-04-16-05-00"},
    {"id": 17, "short_name": "RIV-1", "timestamp": "2024-11-04-16-52-38"},
    {"id": 18, "short_name": "PIL-1", "timestamp": "2024-11-11-12-07-40"},
    {"id": 19, "short_name": "PIL-2", "timestamp": "2024-11-11-12-42-47"},
    {"id": 20, "short_name": "ROOT-1", "timestamp": "2024-11-11-14-29-44"},
    {"id": 21, "short_name": "HAUS-1", "timestamp": "2024-11-11-16-14-23"},
    {"id": 22, "short_name": "HOEB-1", "timestamp": "2024-11-14-11-17-02"},
    {"id": 23, "short_name": "HOEB-2", "timestamp": "2024-11-14-12-01-26"},
    {"id": 24, "short_name": "HEAP-1", "timestamp": "2024-11-14-13-45-37"},
    {"id": 25, "short_name": "KAEB-1", "timestamp": "2024-11-14-14-36-02"},
    {"id": 26, "short_name": "KAEB-2", "timestamp": "2024-11-14-15-22-43"},
    {"id": 27, "short_name": "KAEB-3", "timestamp": "2024-11-14-16-04-09"},
    {"id": 28, "short_name": "TRIM-1", "timestamp": "2024-11-15-10-16-35"},
    {"id": 29, "short_name": "ALB-1", "timestamp": "2024-11-15-11-18-14"},
    {"id": 30, "short_name": "ALB-2", "timestamp": "2024-11-15-11-37-15"},
    {"id": 31, "short_name": "ALB-3", "timestamp": "2024-11-15-12-06-03"},
    {"id": 32, "short_name": "LMB-1", "timestamp": "2024-11-15-14-14-12"},
    {"id": 33, "short_name": "LMB-2", "timestamp": "2024-11-15-14-43-52"},
    {"id": 34, "short_name": "LEE-1", "timestamp": "2024-11-15-16-41-14"},
    {"id": 35, "short_name": "ARC-1", "timestamp": "2024-11-18-12-05-01"},
    {"id": 36, "short_name": "ARC-2", "timestamp": "2024-11-18-13-22-14"},
    {"id": 37, "short_name": "ARC-3", "timestamp": "2024-11-18-13-48-19"},
    {"id": 38, "short_name": "ARC-4", "timestamp": "2024-11-18-15-46-05"},
    {"id": 39, "short_name": "ARC-5", "timestamp": "2024-11-18-16-59-23"},
    {"id": 40, "short_name": "ARC-6", "timestamp": "2024-11-18-17-13-09"},
    {"id": 41, "short_name": "ARC-7", "timestamp": "2024-11-18-17-31-36"},
    {"id": 42, "short_name": "LEICA-1", "timestamp": "2024-11-25-14-57-08"},
    {"id": 43, "short_name": "LEICA-2", "timestamp": "2024-11-25-16-36-19"},
    {"id": 44, "short_name": "SBB-1", "timestamp": "2024-12-03-13-15-38"},
    {"id": 45, "short_name": "SBB-2", "timestamp": "2024-12-03-13-26-40"},
    {"id": 46, "short_name": "CON-1", "timestamp": "2024-12-09-09-34-43"},
    {"id": 47, "short_name": "CON-2", "timestamp": "2024-12-09-09-41-46"},
    {"id": 48, "short_name": "CON-3", "timestamp": "2024-12-09-11-28-28"},
    {"id": 49, "short_name": "CON-4", "timestamp": "2024-12-09-11-53-11"},
]

def get_mission_by_id(mission_id):
    for m in missions:
        if m["id"] == mission_id:
            return m
    raise ValueError(f"Mission ID {mission_id} not found.")

def get_mission_by_shortname(short_name):
    for m in missions:
        if m["short_name"] == short_name:
            return m
    raise ValueError(f"Mission short name '{short_name}' not found.")

def get_mission_by_timestamp(timestamp):
    for m in missions:
        if m["timestamp"] == timestamp:
            return m
    raise ValueError(f"Mission timestamp '{timestamp}' not found.")

# Backwards compatibility getter (using index as ID)
def get_mission_by_index(idx):
    return get_mission_by_id(idx + 1)["timestamp"]

