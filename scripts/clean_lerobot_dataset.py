from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a local LeRobot dataset and write a quality report.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", default="reports/data_quality.json")
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0, help="0 disables the upper bound.")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    report = build_report(dataset, min_frames=args.min_frames, max_frames=args.max_frames)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote quality report: {report_path}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        return 1
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    return 0


def build_report(dataset: Path, min_frames: int, max_frames: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "dataset": str(dataset.resolve()),
        "exists": dataset.exists(),
        "errors": [],
        "warnings": [],
        "features": {},
        "episodes": {},
        "recommended_exclusions": [],
    }
    if not dataset.exists():
        report["errors"].append("dataset path does not exist")
        return report

    meta = dataset / "meta"
    info_path = meta / "info.json"
    if not info_path.exists():
        report["errors"].append("missing meta/info.json")
        return report

    info = _read_json(info_path)
    report["features"] = info.get("features", {})
    _check_required_features(report)

    episodes = _read_episodes(meta)
    lengths = []
    for episode in episodes:
        episode_id = episode.get("episode_index", episode.get("index", len(lengths)))
        length = _episode_length(episode)
        if length is None:
            report["warnings"].append(f"episode {episode_id}: cannot infer frame length")
            continue
        lengths.append(length)
        if length < min_frames:
            report["recommended_exclusions"].append(
                {"episode_index": episode_id, "reason": f"too short: {length} < {min_frames}"}
            )
        if max_frames and length > max_frames:
            report["recommended_exclusions"].append(
                {"episode_index": episode_id, "reason": f"too long: {length} > {max_frames}"}
            )

    report["episodes"] = {
        "count": len(episodes),
        "length_min": min(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "length_mean": mean(lengths) if lengths else None,
    }
    if not episodes:
        report["warnings"].append("no episode metadata found under meta/")
    return report


def _check_required_features(report: dict[str, Any]) -> None:
    features = report["features"]
    for key in ("observation.state", "action"):
        if key not in features:
            report["warnings"].append(f"missing common feature key: {key}")
    image_keys = [key for key in features if key.startswith("observation.images.")]
    if not image_keys:
        report["warnings"].append("no observation.images.* feature found")


def _read_episodes(meta: Path) -> list[dict[str, Any]]:
    for name in ("episodes.jsonl", "episodes.json"):
        path = meta / name
        if path.exists():
            return _read_json_or_jsonl(path)
    return []


def _episode_length(episode: dict[str, Any]) -> int | None:
    if "length" in episode:
        return int(episode["length"])
    if "dataset_from_index" in episode and "dataset_to_index" in episode:
        return int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
    return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

