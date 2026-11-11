"""Điều phối bài lab và ghi tiến độ thực nghiệm."""
import sys
from pathlib import Path
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from eda import explore
from evaluate import evaluate
from train import train
from preprocess import load_config
from tracking import create_run, update_progress, log_experiment
from visualize import visualize
from report import generate_report


# Chạy các bước và ghi thất bại hoặc thành công theo từng lần chạy.
def main():
    run_id = create_run()
    config = load_config()
    step = "PIPELINE"
    try:
        for step, action in (("EDA", explore), ("TRAIN", train), ("EVALUATE", evaluate),
                             ("VISUALIZE", visualize), ("REPORT", generate_report)):
            update_progress(run_id, step, "STARTED")
            action()
            update_progress(run_id, step, "SUCCESS")
        log_experiment(run_id, config)
        update_progress(run_id, "PIPELINE", "SUCCESS")
    except Exception as error:
        update_progress(run_id, step, "FAILED", str(error))
        raise


if __name__ == "__main__":
    main()
