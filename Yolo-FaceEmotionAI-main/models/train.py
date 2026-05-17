import argparse
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


DEFAULT_DATASET_DIR = Path(
    r"D:\28783\Downloads\chrome\archive (3)\FER2013_7emotions_Uniform_Augmented_Dataset"
)
DEFAULT_MODEL = "yolo11n-cls.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train an emotion classification model with YOLO.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET_DIR, help="Dataset root directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Pretrained YOLO classification model.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help='Training device, for example "0" or "cpu".')
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers. Use 0 on Windows if multiprocessing fails.")
    parser.add_argument("--project", default="runs/classify", help="Ultralytics output directory.")
    parser.add_argument("--name", default="fer2013_emotion", help="Training run name.")
    return parser.parse_args()


def validate_dataset(dataset_dir: Path):
    required_dirs = ["train", "validation", "test"]
    missing_dirs = [name for name in required_dirs if not (dataset_dir / name).is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            f"Dataset directory is missing required folders: {missing_dirs}. "
            f"Expected structure: {dataset_dir}\\train|validation|test\\<class_name>\\*.jpg"
        )


def copy_best_weights(results, models_dir: Path):
    best_weight = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weight.is_file():
        print(f"Training finished, but best weights were not found at: {best_weight}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for target_name in ["best.pt", "trained_model.pt"]:
        target_path = models_dir / target_name
        shutil.copy2(best_weight, target_path)
        print(f"Best weights copied to: {target_path}")

    for target_name in [f"best_{timestamp}.pt", f"trained_model_{timestamp}.pt"]:
        target_path = models_dir / target_name
        shutil.copy2(best_weight, target_path)
        print(f"Timestamped weights copied to: {target_path}")


def main():
    args = parse_args()
    dataset_dir = args.data.resolve()
    validate_dataset(dataset_dir)

    model = YOLO(args.model)
    results = model.train(
        data=str(dataset_dir),
        task="classify",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )

    models_dir = Path(__file__).resolve().parent
    copy_best_weights(results, models_dir)


if __name__ == "__main__":
    main()
