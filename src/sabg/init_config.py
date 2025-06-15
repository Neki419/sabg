import argparse
from pathlib import Path
import yaml

DEFAULT_CONFIG = {
    "device": "cpu",
    "model": "maskrcnn_weights.pth",
    "paths": {
        "output_dir": "your_path_to_output",
        "annotations_file_path": "your_path_to_annotations.json",
        "metrics_file_path": "your_path_to_metrics.xlsx"
    },
    "filter": {
        "min_conf": 0.7,
        "min_area": 200,
        "remove_ripped": True,
        "pixel_thr": 0.5
    },
    "background": {
        "initial_ring_radius": 40,
        "patch_radius": 15,
        "angle_step": 20,
        "buffer_px": 2,
        "max_ring_radius": 300,
        "ring_step": 5
    },
    "output": {
        "save_overlay": True,
        "overlay_draw_background": True,
        "overlay_draw_id": True,
        "save_binary_masks": True,
        "binary_mask_draw_id": True,
        "binary_mask_draw_background": True,
        "save_annotations": True
    }
}


def main():
    parser = argparse.ArgumentParser(description="Create default YAML config.")
    parser.add_argument(
        "-d", "--dir", type=str,
        help="Target directory to save config.yaml. Defaults to current directory."
    )
    args = parser.parse_args()

    target_dir = Path(args.dir) if args.dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)

    config_path = target_dir / "config.yaml"

    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True, sort_keys=False)

    print(f"✅ Config saved to {config_path.resolve()}")


if __name__ == "__main__":
    main()
