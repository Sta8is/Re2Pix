# Setup

## Install dependencies
System Requirements:

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver compatible with CUDA 12.6
* Linux x86-64
* glibc>=2.31 (e.g Ubuntu >=22.04)
* Python 3.10

We **HIGHLY** recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/).

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Re2Pix dependencies can be installed with `pip` (requires `python==3.10`):

```shell
uv venv --python 3.10 --allow-existing
uv sync --extra cu126
uv pip install pytorch-lightning
uv pip install scikit-learn
```


Install Hugging Face CLI: 
```shell
uv tool install -U "huggingface_hub[cli]"
```

## Download checkpoints

1. Get a Hugging Face Access Token with Read permission
2. Login: hf auth login
3. Accept the Llama-Guard-3-8B terms.

Download the model checkpoints and tokenizer using the following command:
```shell
hf download \
  nvidia/Cosmos-Predict2-2B-Video2World \
  --revision f50c09f5d8ab133a90cac3f4886a6471e9ba3f18 \
  --local-dir checkpoints/nvidia/Cosmos-Predict2-2B-Video2World \
  --include="model-720p-16fps.pt" \
  --include="tokenizer/*"
```

Download Checkpoints using the following command:

```shell
hf download Sta8is/Re2Pix_cityscapes_40k \
  --local-dir checkpoints/Re2Pix_cs_40k 
```

## Download datasets

### Cityscapes 
Download Cityscapes in mp4 format from [https://huggingface.co/datasets/Sta8is/cityscapes_seq_video](https://huggingface.co/datasets/Sta8is/cityscapes_seq_video) 

```shell
hf download Sta8is/cityscapes_seq_video \
  --repo-type dataset \
  --local-dir benchmark_train/cityscapes_seq_video
```