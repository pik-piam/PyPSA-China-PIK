#!/usr/bin/env python3
# https://github.com/LUMC/slurm-cluster-status/blob/master/README.md
import argparse
import subprocess

STATE_MAP = {
    "BOOT_FAIL": "failed",
    "CANCELLED": "failed",
    "COMPLETED": "success",
    "CONFIGURING": "running",
    "COMPLETING": "running",
    "DEADLINE": "failed",
    "FAILED": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "PENDING": "running",
    "PREEMPTED": "failed",
    "RUNNING": "running",
    "RESIZING": "running",
    "SUSPENDED": "running",
    "TIMEOUT": "failed",
    "UNKNOWN": "running",
}


def fetch_status(batch_id):
    """Fetch the status for the batch id"""
    sacct_args = ["sacct", "-j", batch_id, "-o", "State", "--parsable2", "--noheader"]

    try:
        output = subprocess.check_output(sacct_args).decode("utf-8").strip()
    except Exception:
        # If sacct fails for whatever reason, assume its temporary and return 'running'
        output = "UNKNOWN"

    # Sometimes, sacct returns nothing, in which case we assume it is temporary
    # and return 'running'
    if not output:
        output = "UNKNOWN"

    # The first output is the state of the overall job
    # See
    # https://stackoverflow.com/questions/52447602/slurm-sacct-shows-batch-and-extern-job-names
    # for details
    job_status = output.split("\n")[0]

    # If the job was cancelled manually, it will say by who, e.g "CANCELLED by 12345"
    # We only care that it was cancelled
    if job_status.startswith("CANCELLED by"):
        job_status = "CANCELLED"

    # Otherwise, return the status
    try:
        return STATE_MAP[job_status]
    except KeyError:
        raise NotImplementedError(
            f"Encountered unknown status '{job_status}' when parsing output:\n'{output}'"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_id", type=str)
    args = parser.parse_args()

    status = fetch_status(args.batch_id)
    print(status)
