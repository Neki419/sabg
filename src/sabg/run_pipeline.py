import argparse
import yaml
from pathlib import Path
from sabg.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run cell segmentation pipeline.")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to input image folder.")
    parser.add_argument("-c", "--config_path", required=True, help="Path to YAML configuration file.")
    parser.add_argument("-o", "--output_dir", help="Optional output directory to override config.")
    parser.add_argument("-a", "--annotations_path", help="Optional annotations file path to override config.")
    parser.add_argument("-m", "--metrics_path", help="Optional metrics file path to override config.")

    args = parser.parse_args()

    # Загрузка конфигурации
    config_path = Path(args.config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Переопределение путей, если указаны явно
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.annotations_path:
        cfg["paths"]["annotations_file_path"] = args.annotations_path
    if args.metrics_path:
        cfg["paths"]["metrics_file_path"] = args.metrics_path

    # Запуск пайплайна
    run_pipeline(args.input_dir, cfg)


if __name__ == "__main__":
    main()
