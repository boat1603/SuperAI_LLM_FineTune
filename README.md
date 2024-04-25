## Installation

```bash
ml Mamba
conda create -p ./env python=3.10.0 -y
conda activate ./env
pip install -e .
```

## Submit Train Model

```bash
sbatch submit_multinode.sh
```
Note: 
- Change training config via `./scripts/smultinode.sh`.
- When using Deepspeed training Scheduler will follow the Deepspeed config.
- This code can run in a single node via `#SBATCH -N 1 -c 64`.

## Convert Deepspeed to FP32

```bash
sbatch ./scripts/submit_zero_to_fp32.sh
```