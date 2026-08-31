# Docker environments

## Start

Run from the repository root. Containers run in the foreground and are removed on exit.

x86 NVIDIA CUDA:

```bash
bash ./docker/run.sh sd15
bash ./docker/run.sh flux2
```

Jetson / L4T:

```bash
bash ./docker/run_l4t.sh sd15
bash ./docker/run_l4t.sh flux2
```

Jupyter Lab: `http://localhost:8888/lab`
TensorBoard: `http://localhost:6006/`

## Images

| Service | Dockerfile | Status |
|---|---|---|
| `sd15-cuda` | `Dockerfile.sd15.cuda` | x86 NVIDIA |
| `sd15-l4t` | `Dockerfile.sd15.l4t` | JetPack 6 / L4T r36 |
| `flux2-cuda` | `Dockerfile.flux2.cuda` | x86 NVIDIA |
| `flux2-l4t` | `Dockerfile.flux2.l4t` | Experimental; Orin Nano memory is limited |

`flux2-l4t` is intentionally an **environment-only** image. It checks the
Jetson CUDA/PyTorch/Diffusers stack but does not support LoRA training. See
[`../flux2/README.md`](../flux2/README.md) for the Japanese setup runbook.

The scripts use direct `docker build` and `docker run --rm -it` commands. Compose is not executed by the scripts; commented Compose alternatives and `docker-compose.yml` remain as reference only. One shared `requirements.txt`
contains model-independent packages. CUDA, PyTorch, Diffusers, Transformers and
backend-specific packages are installed by each Dockerfile.

The repository root is the build context so Dockerfiles can copy explicitly
named helpers from `shared/`. The root `.dockerignore` restricts the transmitted
context to `docker/` and `shared/`; datasets, weights, prompts and outputs are
never included in an image build.

## Environment check

The container name follows `ai-image-lab-<backend>-<platform>`.

```bash
docker exec -it ai-image-lab-sd15-l4t check-ai-image-environment
docker exec -it ai-image-lab-flux2-cuda check-ai-image-environment
docker exec -it ai-image-lab-flux2-l4t check-ai-image-environment
```

## TensorBoard

TensorBoard starts for both backends unless disabled.

```bash
ENABLE_TENSORBOARD=0 bash ./docker/run.sh flux2
```

Default log directories:

```text
/workspace/outputs/sd15/training/logs
/workspace/outputs/flux2/training/logs
```

## Hugging Face token

```bash
HF_TOKEN="..." bash ./docker/run.sh flux2
```

The token is passed at runtime and is never written to an image or repository.
