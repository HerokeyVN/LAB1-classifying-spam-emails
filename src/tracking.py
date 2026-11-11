"""Ghi tiến độ, nhật ký và cấu hình thực nghiệm."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


# Tạo mã lần chạy và các thư mục nhật ký.
def create_run():
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    for name in ("logs", "progress_results"):
        (ROOT / name).mkdir(exist_ok=True)
    update_progress(run_id, "PIPELINE", "STARTED")
    return run_id


# Ghi trạng thái mới nhất và một dòng nhật ký có thời gian UTC.
def update_progress(run_id, step, status, error=None):
    record = dict(run_id=run_id, step=step, status=status,
                  timestamp=datetime.now(timezone.utc).isoformat(), error=error)
    (ROOT / "progress_results" / f"{run_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / "logs" / f"{run_id}.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


# Lưu metrics thực tế và cấu hình của lần chạy thành công.
def log_experiment(run_id, config):
    with (ROOT / "reports" / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    path = ROOT / "docs" / "experiment_logs.csv"
    fields = ["run_id", "finished_at", "model", "accuracy", "precision", "recall", "f1", "config_json"]
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(dict(run_id=run_id, finished_at=datetime.now(timezone.utc).isoformat(),
                model=row["Model"], accuracy=row["Accuracy"], precision=row["Precision"],
                recall=row["Recall"], f1=row["F1-Score"], config_json=json.dumps(config)))
