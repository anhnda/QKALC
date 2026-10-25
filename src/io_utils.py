from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_json(path: str | Path, payload: Any, *, indent: int = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + '.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=indent)
        temp_path.replace(path)
        return
    except OSError:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass

    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=indent)