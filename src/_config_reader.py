__all__ = [
    "ALLOW_VISUALS",
    "DEBUG_MODE"
]

import json
import project_paths

config_file_path = project_paths.base / "configuration.json"

with open(str(config_file_path), 'r') as f:
    data = json.load(f)
    
ALLOW_VISUALS : bool = data["allow_visuals"]
DEBUG_MODE : bool = data["debug_mode"]