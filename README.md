# Only Train Once: Uncertainty-Aware One-Class Learning for Face Authenticity Detection

This repository contains the official implementation of **“Only Train Once: Uncertainty-Aware One-Class Learning for Face Authenticity Detection.”** The paper has been accepted for publication in *IEEE Transactions on Multimedia (TMM)*.
<img width="830" height="289" alt="image" src="https://github.com/user-attachments/assets/739dc22c-240b-41f0-94d9-e2f5b9c4b6e8" />
The source code, complete pretrained model, and evaluation configuration used in this release are publicly available. The evaluation scripts report numerical metrics and save results as JSON files.

## Contents

- `fadnet.py`, `metrics.py`: FADNet definition, data loading, and metric evaluation.
- `test_commercial_tools.py`: evaluation on DALL-E 2, IF, and Midjourney.
- `test_df40.py`: per-method evaluation on DF40.
- `test_df40_mixed.py`: evaluation on the combined DF40 sources.
- `test_asfd.py`: evaluation on ASFD generators.
- `test_custom_dataset.py`: evaluation on a user-specified real/synthetic image pair.
- `models/fadnet.pth`: complete pretrained FADNet checkpoint.

## Requirements

`requirements.txt` records the pinned Python packages from the `facedetection` environment (Python 3.10.19). The PyTorch packages match its Windows CUDA 12.9 setup.

```
pip install -r requirements.txt

```

## Dataset Sources

- [Asian Synthetic Face Dataset (ASFD)](https://github.com/Hurrice-star/ASFD)
- [DF40](https://github.com/YZY-stack/DF40)

Please refer to the dataset repositories for access instructions and usage terms. 

## Evaluation

Run the scripts from this directory. Dataset paths can be supplied through command-line arguments.

```bash
python test_commercial_tools.py --real /path/to/real --fake dalle2=/path/to/dalle2 --fake if=/path/to/if --fake midjourney=/path/to/midjourney
python test_df40.py --real /path/to/real --df40-root /path/to/DF40
python test_df40_mixed.py --real /path/to/real --df40-root /path/to/DF40
python test_asfd.py --real /path/to/real --fake ProGAN=/path/to/ProGAN
python test_custom_dataset.py --real /path/to/real --fake /path/to/synthetic
```

Each script accepts `--model`, `--output`, and `--batch-size`. The commercial and DF40 evaluations use the fixed uncertainty threshold `0.6911`. ASFD reproduces the per-dataset threshold search used in the original evaluation by default; pass `--fixed-threshold` to use `0.6911` instead. Results are saved as JSON under `results/` by default.
