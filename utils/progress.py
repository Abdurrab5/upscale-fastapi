import json
import os

PROGRESS_DIR = "progress"

os.makedirs(PROGRESS_DIR, exist_ok=True)


def update_progress(job_id: str, percent: int, message: str):

    with open(
        os.path.join(PROGRESS_DIR, f"{job_id}.json"),
        "w"
    ) as f:

        json.dump(
            {
                "percent": percent,
                "message": message
            },
            f
        )


def get_progress(job_id: str):

    path = os.path.join(
        PROGRESS_DIR,
        f"{job_id}.json"
    )

    if not os.path.exists(path):
        return {
            "percent": 0,
            "message": "Waiting"
        }

    with open(path) as f:
        return json.load(f)


def remove_progress(job_id: str):

    path = os.path.join(
        PROGRESS_DIR,
        f"{job_id}.json"
    )

    if os.path.exists(path):
        os.remove(path)