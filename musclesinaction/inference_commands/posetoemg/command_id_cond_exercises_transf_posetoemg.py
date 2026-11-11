"""Launch pose-to-EMG evaluation in explicitly addressed tmux sessions."""
import argparse
from pathlib import Path
import shlex
import shutil
import subprocess


EXERCISES = [
    "SlowSkater", "Running", "RonddeJambe", "LegCross", "LegBack",
    "KneeKick", "JumpingJack", "HookPunch", "HighKick", "FrontPunch",
    "FrontKick", "ElbowPunch", "Shuffle", "SideLunges", "Squat",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="Defaults to CUDA using musclesinaction-gpu; CPU uses the legacy environment.")
    parser.add_argument("--exercises", nargs="+", choices=EXERCISES, default=EXERCISES)
    parser.add_argument("--env", default=None, help="Conda environment (default: musclesinaction-gpu for CUDA, musclesinaction for CPU)")
    args = parser.parse_args()
    environment = args.env or ("musclesinaction-gpu" if args.device == "cuda" else "musclesinaction")
    root = Path(__file__).resolve().parents[3]
    conda = shutil.which("conda")
    if conda is None:
        parser.error("conda must be on PATH")
    for exercise in dict.fromkeys(args.exercises):
        session = "mia_posetoemg_" + ("gpu_" if args.device == "cuda" else "") + exercise
        if subprocess.run(["tmux", "has-session", "-t", session],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            print("Session already exists, skipping: " + session, flush=True)
            continue
        name = "generalization_new_cond_" + exercise + "_clean_baseline_perex"
        if args.device == "cuda":
            name += "_gpu"
        log_dir = root / "logs" / name
        log_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=0",
            conda, "run", "--no-capture-output", "-n", environment,
            "python", "-u", "-m",
            "musclesinaction.inference_scripts.inference_id_transf_cond_exercises_posetoemg",
            "--name", name, "--device", args.device,
            "--std", "False", "--threed", "True", "--predemg", "True", "--cond", "True",
            "--resume", "pretrained-checkpoints/generalization_new_cond_clean_posetoemg/model_100.pth",
            "--data_path_train", "musclesinaction/ablation/generalization_ID_nocond_exercises/train.txt",
            "--data_path_val", "musclesinaction/ablation/generalization_ID_nocond_exercises/val" + exercise + ".txt",
            "--log_path", str(log_dir),
        ]
        # Start the process directly: send-keys without -t can target the caller's pane.
        # Retain the pane on completion so errors and results remain inspectable.
        shell_command = ("tmux set-option -w -t " + shlex.quote(session)
                         + " remain-on-exit on; exec " + shlex.join(command)
                         + " > " + shlex.quote(str(log_dir / "console.log")) + " 2>&1")
        subprocess.run(["tmux", "new-session", "-d", "-s", session,
                        "-c", str(root), shell_command], check=True)
        print("Started " + session + "; log: " + str(log_dir / "console.log"), flush=True)


if __name__ == "__main__":
    main()
