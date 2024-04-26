## Installation

Repository preparation

```bash
git clone https://github.com/boat1603/SuperAI_LLM_FineTune.git
cd ./SuperAI_LLM_FineTune
```

### Install using Conda

```bash
ml Mamba
conda create -p ./env python=3.10.0 -y
conda activate ./env
pip install -e .
```

### Install using Apptainer (Optional)

```bash
ml Apptainer
apptainer build ./llm-finetune.sif docker://boat1603/llm-finetune:latest
```

## Submit Train Model

```bash
sbatch submit_multinode.sh
```

for Apptainer
```bash
sbatch submit_multinode_apptainer.sh
```

Note:

- Change training config via `./smultinode.sh` or `./smultinode_apptainer.sh` (for apptainer).
- When using Deepspeed training Scheduler will follow the Deepspeed config.
- You can setup training spec in `./submit_multinode.sh` or `submit_multinode_apptainer.sh` following [our guideline](https://openthaigpt.gitbook.io/openthaigpt-guideline/lanta/slurm).

## Convert Deepspeed to FP32

```bash
sbatch ./submit_zero_to_fp32.sh
```
