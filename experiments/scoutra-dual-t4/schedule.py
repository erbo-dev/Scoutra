"""One spawn process per device, with a shared queue and explicit failure recovery."""
import argparse
import contextlib
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import time
import traceback
from pathlib import Path

from common import Blocked, json_read, json_write, table
from jobs import JobRunner

SAMPLE_FIELDS = ("utc_epoch_s", "physical_index", "gpu_uuid", "utilization_pct", "memory_used_mib", "memory_total_mib")


def worker(device, first, work, events, log_path, factory, expected_uuid):
    pid = os.getpid()
    base = {"device": device, "pid": pid}
    last = time.monotonic()
    with open(log_path, "w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            runtime_device = device
            if device.startswith("cuda:"):
                logical = int(device.split(":")[1])
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                token = visible.split(",")[logical].strip() if visible else str(logical)
                os.environ["CUDA_VISIBLE_DEVICES"] = f"GPU-{expected_uuid}" if expected_uuid else token
                runtime_device = "cuda:0"
            runner = factory(runtime_device)
            physical_uuid = getattr(runner, "gpu_uuid", None)
            if expected_uuid and physical_uuid != expected_uuid:
                raise RuntimeError(f"{device} worker resolved GPU {physical_uuid}, expected {expected_uuid}")
            if physical_uuid:
                base["gpu_uuid"] = physical_uuid
        except Exception:
            events.put(("startup_failure", dict(base, traceback=traceback.format_exc())))
            return
        idle = 0.0
        task = first
        while task is not None:
            start = time.monotonic()
            idle += start - last
            row = dict(base, id=task["id"], method=task["method"], clip=task.get("video", {}).get("clip", ""),
                       phase=task["phase"], start_monotonic_s=start)
            events.put(("start", row))
            try:
                metrics = runner({**task, "assigned_device": device})
                result = dict(row, state="EXECUTED", **metrics)
            except Exception as error:
                print(traceback.format_exc(), flush=True)
                state = "BLOCKED" if isinstance(error, Blocked) else "FAILED"
                result = dict(row, state=state, reason=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            last = time.monotonic()
            events.put(("result", dict(result, wall_s=last - start, end_monotonic_s=last)))
            task = work.get()
        events.put(("worker_done", dict(base, idle_estimate_s=idle + time.monotonic() - last)))


def sample_devices(stop, samples, diagnostics):
    while not stop.is_set():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if result.returncode:
                diagnostics.append(result.stderr.strip())
                return
            stamp = time.time()
            for line in result.stdout.splitlines():
                columns = [part.strip() for part in line.split(",")]
                if len(columns) == 5:
                    samples.append(dict(zip(SAMPLE_FIELDS, [stamp, *columns])))
        except (OSError, subprocess.TimeoutExpired) as error:
            diagnostics.append(str(error))
            return
        stop.wait(1)


def execute(tasks, devices, logdir, timeout, factory=JobRunner, telemetry=True, device_uuids=None):
    if not tasks or not devices:
        raise ValueError("Scheduler requires nonempty jobs and device plan")
    if len({job["id"] for job in tasks}) != len(tasks) or len(set(devices)) != len(devices):
        raise ValueError("Scheduler jobs and devices must have unique identifiers")
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    devices = devices[:len(tasks)]
    sorted_jobs = sorted(tasks, key=lambda job: (-job.get("cost", 0), job["id"]))
    context = mp.get_context("spawn")
    work, events = context.Queue(), context.Queue()
    processes = {}
    started = {}
    final = {}
    workers = {}
    events_by_id = {}
    samples, diagnostics = [], []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_devices, args=(stop, samples, diagnostics), daemon=True)
    submissions = time.monotonic()
    if telemetry:
        sampler.start()

    def receive(message):
        kind, event = message
        pid = event["pid"]
        if kind == "start":
            started[pid] = event
            events_by_id[event["id"]] = event
            print(f"[{event['device']} | pid {pid}] {event['phase']} {event['method']} {event['clip']}", flush=True)
        elif kind == "result":
            started.pop(pid, None)
            final[event["id"]] = event
        elif kind in {"startup_failure", "worker_done"}:
            workers[pid].update(event)
            workers[pid]["terminal"] = kind

    try:
        # Seed one job per worker, then let available workers pull remaining jobs.
        for device, first in zip(devices, sorted_jobs):
            tag = device.replace(":", "_")
            log = logdir / f"{tag}.log"
            process = context.Process(target=worker, args=(device, first, work, events, str(log), factory, (device_uuids or {}).get(device)))
            process.start()
            processes[process.pid] = process
            workers[process.pid] = {"device": device, "pid": process.pid, "seed_job": first["id"], "log": str(log),
                                    "spawned": time.monotonic()}
        for job in sorted_jobs[len(devices):]:
            work.put(job)
        for _ in devices:
            work.put(None)

        while any(process.is_alive() for process in processes.values()):
            try:
                receive(events.get(timeout=0.15))
            except queue.Empty:
                pass
            now = time.monotonic()
            for pid, process in processes.items():
                worker_info = workers[pid]
                active = started.get(pid)
                age = now - (active["start_monotonic_s"] if active else worker_info["spawned"])
                if process.is_alive() and age > timeout and (active or worker_info["seed_job"] not in final):
                    worker_info["timeout"] = f"Worker exceeded {timeout}s for active job or process initialization"
                    process.terminate()
                    process.join(timeout=4)
                    if process.is_alive():
                        process.kill()
        for process in processes.values():
            process.join()
        while True:
            try:
                receive(events.get(timeout=.1))
            except queue.Empty:
                break
    finally:
        end = time.monotonic()
        stop.set()
        if telemetry:
            sampler.join(timeout=6)
        for pid, process in processes.items():
            if process.is_alive():
                process.kill()
                process.join(timeout=4)
            workers[pid]["exit_code"] = process.exitcode
        work.cancel_join_thread()
        work.close()
        events.close()

    initial = {info["seed_job"]: info for info in workers.values()}
    for job in tasks:
        if job["id"] in final:
            continue
        row = events_by_id.get(job["id"], {})
        assigned = workers.get(row.get("pid"), initial.get(job["id"]))
        reason = "All available workers stopped before returning this job"
        if assigned:
            lines = assigned.get("traceback", "").strip().splitlines()
            reason = assigned.get("timeout") or (lines[-1] if lines else f"Worker exited with code {assigned['exit_code']}")
        final[job["id"]] = {
            **row, "id": job["id"], "method": job["method"], "clip": job.get("video", {}).get("clip", ""),
            "phase": job["phase"], "state": "FAILED" if assigned else "BLOCKED", "reason": reason,
            "device": assigned["device"] if assigned else "", "pid": assigned["pid"] if assigned else "",
        }
    results = [final[job["id"]] for job in tasks]
    for pid, info in workers.items():
        successful = [row for row in results if row.get("pid") == pid and row["state"] == "EXECUTED"]
        info.update(frames=sum(r.get("frames", 0) for r in successful), successful_jobs=len(successful),
                    cuda_frames=sum(r.get("cuda_frames", 0) for r in successful),
                    inference_batches=sum(r.get("inference_batches", 0) for r in successful),
                    inference_s=sum(r.get("inference_s", 0) for r in successful),
                    failures=sum(r.get("pid") == pid and r["state"] != "EXECUTED" for r in results))
    if samples:
        table(logdir / "device_samples.csv", samples, SAMPLE_FIELDS)
    return dict(results=results, workers=list(workers.values()), wall_s=end - submissions,
                telemetry_samples=samples, telemetry_diagnostics=diagnostics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    spec = json_read(args.spec)
    result = execute(spec["tasks"], spec["devices"], spec["logdir"], spec["timeout"], telemetry=spec["telemetry"], device_uuids=spec.get("device_uuids"))
    json_write(args.result, result)


if __name__ == "__main__":
    main()
