#!/usr/bin/env python3
"""Build a machine-local configuration from the portable release manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = (
    "input_video_path", "gt_video_path", "glyph_mask_path",
    "bbox_json_path", "bbox_mask_video_path", "mask_frame_root",
)


def repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="extracted dataset root")
    parser.add_argument("--prediction-root", required=True, help="method output root")
    parser.add_argument("--model", default="your_method")
    parser.add_argument("--setting", choices=("blind", "mask-guided"), default="mask-guided")
    parser.add_argument("--geometry", choices=("full_frame_resize_to_gt", "clear_center_crop_inverse"),
                        default="full_frame_resize_to_gt")
    parser.add_argument("--temporal", choices=("exact", "verified_prefix"), default="verified_prefix")
    parser.add_argument("--manifest", default="manifests/benchmark_manifest.json")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--config-out", default="configs/local.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    prediction_root = Path(args.prediction_root).expanduser().resolve()
    if not (data_root / "subtitled_data").is_dir():
        parser.error("--data-root must contain the subtitled_data directory")
    if not prediction_root.is_dir():
        parser.error("--prediction-root must be an existing directory")
    if not args.model.strip() or args.model == "all":
        parser.error("--model must be a nonempty name other than 'all'")

    template = ROOT / "configs/evaluation.example.json"
    manifest_path = repository_path(args.manifest)
    output_root = repository_path(args.output_root)
    config_path = repository_path(args.config_out)
    resolved_path = output_root / "resolved_manifest.json"
    if config_path in (template, manifest_path) or resolved_path in (template, manifest_path) or config_path == resolved_path:
        parser.error("runtime outputs must not overwrite the public inputs or each other")
    for path in (config_path, resolved_path):
        if path.exists() and not args.overwrite:
            parser.error(f"{path} exists; use a new output path or --overwrite")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest if isinstance(manifest, list) else manifest.get("samples")
    if not isinstance(rows, list):
        parser.error("manifest must be a list or an object with a samples list")
    for row in rows:
        for field in PATH_FIELDS:
            if row.get(field):
                path = Path(row[field])
                row[field] = str(path if path.is_absolute() else data_root / path)

    config = json.loads(template.read_text(encoding="utf-8"))
    config["manifest"] = str(resolved_path)
    config["input_root"] = str(data_root / "subtitled_data")
    config["output_root"] = str(output_root / "evaluation")
    config["runtime"]["torch_home"] = str(output_root / "torch_cache")
    config["models"] = {
        args.model: {
            "setting": args.setting,
            "result_root": str(prediction_root),
            "geometry": args.geometry,
            "temporal": args.temporal,
        }
    }
    for name, profile in config["evaluation_profiles"].items():
        profile["output_root"] = str(output_root / name)
    config["analysis"]["metadata_path"] = str(ROOT / "manifests/sample_metadata.json")
    config["analysis"]["output_dir"] = str(output_root / "analysis")
    for path, value in ((resolved_path, manifest), (config_path, config)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "config": str(config_path), "manifest": str(resolved_path),
        "sample_count": len(rows), "model": args.model,
    }, indent=2))


if __name__ == "__main__":
    main()
