# Long Video Token Dataset Distillation

Hello! Thanks for checking out our code supplement. This archive contains the implementation used for our long-video token dataset distillation experiments.

## Setup

1. Enter the code directory:

```bash
cd Lite-VD
```

2. Install the required environment:

```bash
pip install -r requirements.txt
```

3. Datasets preparation.

For UCF101 and HMDB51, you can use [mmaction2](https://github.com/open-mmlab/mmaction2) to extract the raw frame and then resize them using [resize_mydata.py](./distill_utils/resize_mydata.py). For Something-Something V2, you can extract frames using the code in [extract_frames/](./extract_frames/) which were modified from [video_distillation](https://github.com/yuz1wan/video_distillation).

Put the prepared datasets under:

```text
distill_utils/data/
  UCF101/
  HMDB51/
  SSv2/
```

The expected files are:

```text
distill_utils/data/UCF101/
  ucf101_splits1.csv
  jpegs_112/

distill_utils/data/HMDB51/
  hmdb51_splits.csv
  jpegs_112/

distill_utils/data/SSv2/
  annot_train.json
  annot_val.json
  frame/       # or rawframes/
```

Please remember to turn off preload for large-scale datasets such as SSv2 by setting `PRELOAD=0`.

4. Prepare quantized VAE weights.

Please download the quantized model weights from [Hugging Face](https://huggingface.co/datasets/Ning9319).

```text
quantized_vae/quantized_vae.safetensors
quantized_3dvae/quantized_3dvae.pt
```

The corresponding config and preparation files are included. The 2D VAE path uses `quantize_vae.py`; the 3D VAE path uses `dquantize_3dvae.py`.

5. Optional: disable online wandb logging.

```bash
export WANDB_MODE=disabled
```

## Running

The main command format is:

```bash
# bash main.sh GPU_ID Dataset IPC
bash main.sh 0 miniUCF101 24
```

We use 500 downstream training epochs and 5 random evaluation runs for the standard setting:

```bash
NUM_EVAL=5
EPOCH_EVAL_TRAIN=500
EVAL_TEST_FREQ=500
```

Below is the standard command template:

```bash
FRAMES=16 \
PRELOAD=1 \
VAE_MODEL=2DVAE \
METHOD=ImportanceHOSVD \
EVAL_MODE=top5 \
EVAL_MODELS=VideoMAE \
NUM_EVAL=5 \
EPOCH_EVAL_TRAIN=500 \
EVAL_TEST_FREQ=500 \
BATCH_TRAIN=128 \
TEST_BATCH_SIZE=8 \
ENCODE_BATCH_SIZE=16 \
VIDEO_TRANSFORMER_TUNE_MODE=full_finetune \
bash main.sh 0 miniUCF101 24
```

For HMDB51, replace the dataset name:

```bash
bash main.sh 0 HMDB51 24
```

For SSv2, use 8 frames and disable preload:

```bash
FRAMES=8 PRELOAD=0 bash main.sh 0 SSv2 24
```

In our main memory-budgeted latent experiments, `ImportanceHOSVD` and `LVDD_Tucker` use `IPC=24`. Pixel-space baselines use the conventional pixel IPC setting, so `Random`, `Herding`, and `DM` use `IPC=1`.

To run other methods, change `METHOD` and use the corresponding IPC:

```bash
METHOD=LVDD_Tucker LVDD_SELECT_MODE=DAPS bash main.sh 0 miniUCF101 24
METHOD=Random bash main.sh 0 miniUCF101 1
METHOD=Herding bash main.sh 0 miniUCF101 1
METHOD=DM bash main.sh 0 miniUCF101 1
```

Here, `Random`, `Herding`, and `DM` are pixel-space baselines, while `LVDD_Tucker` is the latent baseline.

The commonly used method names are:

```text
ImportanceHOSVD
LVDD_Tucker
Random
Herding
DM
Full
```

## Saved Artifacts

Distilled results are saved under:

```text
logged_files/LongVideoToken_<DATASET>_<METHOD>_<VAE_MODEL>/<DATASET>_ipc<IPC>_<TIME>/
```

Important output files include:

```text
synthetic_data.pt
distill_report.json
distill_state.pt
```

To evaluate an existing latent artifact directly, pass `synthetic_data.pt` as the fourth argument:

```bash
FRAMES=16 \
PRELOAD=1 \
VAE_MODEL=2DVAE \
METHOD=ImportanceHOSVD \
EVAL_MODE=top5 \
EVAL_MODELS=VideoMAE \
NUM_EVAL=5 \
EPOCH_EVAL_TRAIN=500 \
EVAL_TEST_FREQ=500 \
BATCH_TRAIN=128 \
TEST_BATCH_SIZE=8 \
ENCODE_BATCH_SIZE=16 \
VIDEO_TRANSFORMER_TUNE_MODE=full_finetune \
bash main.sh 0 miniUCF101 24 logged_files/.../synthetic_data.pt
```

## Visualization

Visualize decoded distilled samples:

```bash
python show_img.py \
  --artifact logged_files/.../synthetic_data.pt \
  --vae_model 2DVAE \
  --class_id 1 \
  --num_videos 1 \
  --save_dir paper_ours_visualizations/example
```

Compare baseline and our distilled frames:

```bash
python compare_method_frames.py \
  --baseline_artifact logged_files/.../baseline/synthetic_data.pt \
  --ours_artifact logged_files/.../ours/synthetic_data.pt \
  --vae_model 2DVAE \
  --class_id 1 \
  --rank 0 \
  --save_dir paper_ours_visualizations/example_compare
```

## Workflow

The code follows this pipeline:

1. Load video frames from `distill_utils/data/`.
2. Encode videos into VAE latent tokens and cache them under `latent_cache/`.
3. Select representative videos with the specified method.
4. Compress selected latent tokens with HOSVD.
5. Decode the distilled set and evaluate it with VideoMAE.

## Notes

- `VIDEO_TRANSFORMER_TUNE_MODE=full_finetune` trains the full VideoMAE model and is used for the standard full downstream evaluation setting.
- `VIDEO_TRANSFORMER_TUNE_MODE=linear_probe` freezes the VideoMAE backbone and can be used for faster debugging runs.
- For memory-budgeted comparisons, compare `stored_artifact_mb` in `distill_report.json`, not only the nominal IPC value.

## Acknowledgements

Our implementation builds on ideas and utilities from:

- Latent Video Dataset Distillation
- Dancing with Still Images: Video Distillation via Static-Dynamic Disentanglement
- CV-VAE: A Compatible Video VAE for Latent Generative Video Models
- MMAction2
- Hugging Face Transformers and Diffusers

## Reference

If you find this code useful, please cite our paper.
