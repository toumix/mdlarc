"""
Run mdlARC on Modal: the dataset is built into the image exactly as the README
builds it, training and inference run `run_script.py` on one GPU.

    modal run modal_app.py::audit
    modal run modal_app.py::train --preset smoke
    modal run --detach modal_app.py::train --preset high --claim high-1
    modal run modal_app.py::fetch --name high-1

`--claim` names the line of `EVALUATION.md` the run scores; without it the
run never reads `assets/solutions.json`. Everything lands on the volume
`mdlarc`; results come back as return values because `modal volume get` is
blocked from the agents' sandbox.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal

# torch 2.8's inductor breaks the compiled bf16 forward (an fp32/bf16 addmm), and flash-attn
# ships no wheel past torch 2.8, so it is built from source for the H100 (sm_90) only.
TORCH = "2.10.0"
FLASH = "2.8.3"
ROOT = "/mdlarc"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install(f"torch=={TORCH}", "numpy", "numba", "matplotlib", "packaging", "ninja",
                 "psutil", "wheel", extra_index_url="https://download.pytorch.org/whl/cu128")
    .env({"FLASH_ATTN_CUDA_ARCHS": "90", "MAX_JOBS": "8", "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
          # the image's Python was built with clang, which the CUDA image lacks
          "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared"})
    .run_commands(f"pip install flash-attn=={FLASH} --no-build-isolation")
    .add_local_dir("dataset_building_scripts", f"{ROOT}/dataset_building_scripts", copy=True)
    .run_commands(
        f"cd {ROOT}/dataset_building_scripts && python download_and_group.py"
        " && python build_datasets.py arc1 --add-conceptarc --with-filtered",
        f"cd {ROOT} && sha256sum assets/challenges.json assets/solutions.json > assets/SHA256")
    .add_local_dir("src", f"{ROOT}/src")
    .add_local_file("run_script.py", f"{ROOT}/run_script.py")
    .add_local_file("audit.py", f"{ROOT}/audit.py")
)
app = modal.App("mdlarc", image=image)
volume = modal.Volume.from_name("mdlarc", create_if_missing=True)
VOL = "/vol"


@app.function(timeout=600)
def audit() -> str:
    """Content-level leakage audit of the training build against ARC-AGI-1 evaluation."""
    tmp = f"{ROOT}/assets_tmp"
    report = subprocess.run(
        ["python", "-I", f"{ROOT}/audit.py", f"{tmp}/ARC-1/raw", f"{tmp}/ARC-2/raw",
         f"{tmp}/ConceptARC/raw"],
        capture_output=True, text=True, check=True).stdout
    print(report)
    return report


@app.function(gpu=os.environ.get("MDLARC_GPU", "H100"), timeout=6 * 3600, volumes={VOL: volume})
def train(preset: str = "smoke", claim: str = "", name: str = "") -> dict:
    """`run_script.py <preset>`, its log, timing and submission saved under `runs/<name>`."""
    name = name or claim or preset
    out = Path(VOL) / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MDLARC_SCORE": "1" if claim else "0"}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    start = time.time()
    with open(out / "log.txt", "w") as log:
        proc = subprocess.run(["python", "run_script.py", preset], cwd=ROOT, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    seconds = time.time() - start
    runs = Path(ROOT) / "runs"
    for sub in runs.glob("*/submission.json"):
        shutil.copy(sub, out / "submission.json")
        shutil.copy(sub.with_name("candidates.json"), out / "candidates.json")
    shutil.copytree(runs, out / "runs", dirs_exist_ok=True)
    meta = {"preset": preset, "claim": claim, "gpu": gpu, "seconds": round(seconds, 1),
            "returncode": proc.returncode,
            "assets": (Path(ROOT) / "assets" / "SHA256").read_text()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    volume.commit()
    print(json.dumps(meta, indent=1))
    return {**meta, "log_tail": (out / "log.txt").read_text()[-6000:]}


@app.function(volumes={VOL: volume})
def fetch(name: str, path: str = "") -> dict:
    """The files of a run, or one of them."""
    out = Path(VOL) / "runs" / name
    files = [out / path] if path else sorted(f for f in out.iterdir() if f.suffix in (".txt", ".json"))
    found = {f.name: f.read_text()[-20000:] for f in files if f.is_file() and f.name != "candidates.json"}
    for key, text in found.items():
        print(f"== {key}\n{text}")
    return found


@app.local_entrypoint()
def main(preset: str = "smoke", claim: str = ""):
    print(json.dumps(train.remote(preset, claim), indent=1))
