# Local Environment Setup

This repository includes `environment.yml`, a portable export of the local
`musclesinaction` Conda environment.

## Create the environment

```bash
conda env create --file environment.yml
conda activate musclesinaction
```

If an environment with this name already exists, update it instead:

```bash
conda env update --name musclesinaction --file environment.yml
conda activate musclesinaction
```

The environment uses Python 3.8.16 and the project's legacy ML stack:

- PyTorch 1.4.0
- torchvision 0.5.0
- TensorFlow 2.2.0
- NumPy 1.21.0
- SciPy 1.4.1
- OpenCV, pandas, scikit-learn, SMPL-X, pyrender, trimesh, and wandb

The environment export includes `protobuf==3.20.3`, which is required for
TensorFlow 2.2.0 to import successfully with the installed generated protobuf
files.

## Historical dependency notes

The original `requirements.txt` is a historical Conda export. Three entries
could not be installed from current package indexes:

- `multi-person-tracker==0.1`
- `yolov3==0.1`
- `chumpy==0.71`

The exported environment omits the first two packages, which are not imported
by the core inference/training paths, and uses the available `chumpy==0.70`.

## Verify the installation

From the repository root:

```bash
python -m pip check
python -c "import torch, torchvision, tensorflow, numpy, scipy, cv2, pandas, sklearn, smplx, pyrender, trimesh, yaml, wandb; print('core imports OK')"
```

The code can also be syntax-checked with:

```bash
python -m compileall -q musclesinaction
```

## Inference

Download and place the dataset as described in the project README, then run
one of the provided inference commands. For example:

```bash
python musclesinaction/inference_commands/emgtopose/command_id_cond_exercises_transf_emgtopose.py
```

Check the relevant configuration and checkpoint paths before running a
different inference command.

## GPU pose-to-EMG inference on RTX 50-series

The legacy Python 3.8 / PyTorch 1.4 environment cannot run the model's
convolution on the local RTX 5070 Ti. Use the separate GPU environment:

```bash
conda env create --file environment-gpu.yml
conda activate musclesinaction-gpu
python musclesinaction/inference_commands/posetoemg/command_id_cond_exercises_transf_posetoemg.py --exercises Running
```

CUDA is the launcher's default device. Omit `--exercises Running` to launch all 15 exercise splits. Each job uses
`conda run -n musclesinaction-gpu` and a session named
`mia_posetoemg_gpu_<exercise>`. Logs are stored under
`logs/generalization_new_cond_<exercise>_clean_baseline_perex_gpu/`.
Completed panes remain available, and existing sessions are skipped.
To rerun a completed job without deleting its session, use, for example,
`tmux respawn-pane -t mia_posetoemg_gpu_Running`.

This environment supports this inference path; the original `musclesinaction`
environment retains the full legacy training and rendering dependencies.

## Training

Update the dataset and output paths in
`musclesinaction/configs/train.yaml`, then run:

```bash
python musclesinaction/train.py
```

The training configuration controls whether the model predicts EMG or pose,
the data split, checkpoint directory, device, and other training options.

PyTorch 1.4.0 is a legacy release. GPU training requires a compatible NVIDIA
driver and CUDA runtime; otherwise set the configuration device to CPU.
