from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests import _baseline_v3_common as common


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync existing snapshot SQLite files into the organized dataset catalog.")
    parser.parse_args()
    payload = common.sync_existing_snapshot_catalog()
    print(json.dumps(payload, indent=2, sort_keys=True, default=common._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
