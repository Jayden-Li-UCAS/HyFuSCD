This is the code and processed datasets for "HyFuSCD: A Multi-Dimensional Feature Fusion Network with Hybrid Architecture for Remote Sensing Semantic Change Detection". The associated paper is currently under review, and the full content will be made publicly available promptly upon acceptance.

## Installation
1. **Create a new Conda environment:**

```bash
conda create -n HyFuSCD python=3.9 -y
conda deactivate
conda activate HyFuSCD
pip install -r requirements.txt
```
2.Download the following two files from the specified link:
[[`causal_conv1d`](https://huggingface.co/datasets/Jayden-Li/Standardized_Benchmark_Datasets_for_Remote_Sensing_Semantic_Change_Detection/blob/main/causal_conv1d-1.1.3.post1%2Bcu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl)]
[[`mamba_ssm`](https://huggingface.co/datasets/Jayden-Li/Standardized_Benchmark_Datasets_for_Remote_Sensing_Semantic_Change_Detection/blob/main/mamba_ssm-1.1.3.post1%2Bcu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl)]
```bash
pip install causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install mamba_ssm-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
```
