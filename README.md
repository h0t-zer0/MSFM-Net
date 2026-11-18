# Mamba-based Spatial-Frequency Modeling for RGB-IR Camouflaged Object Detection

This repository accompanies our work on RGB-IR camouflaged object detection. The method uses Mamba-based spatial-frequency modeling to combine visible and infrared information for camouflaged object detection.

## Project status

This work is associated with a pending patent application. To protect the intellectual property related to this work, the complete source code and corresponding experimental results will be made publicly available in this repository after the patent application has been filed. Filing is currently expected within the next 1–2 months.

We plan to release the benchmark, along with ResNet-based model weights and saliency maps for two datasets. Availability and download instructions will be added here when these materials are ready.

## Repository contents

- `Model.py`: model wrapper and network interface.
- `models/`: model and backbone implementations.
- `dataloader.py`: RGB-IR data loading and preprocessing.
- `train.py`: training and validation script.
- `test.py`: saliency map inference script.
- `eval.py`: saliency map evaluation script.
- `mssim.py`: structural similarity loss implementation.


## Data organization

The training and testing scripts expect each dataset split to contain aligned RGB images, thermal infrared images, and ground-truth masks:

```text
<dataset split>/
├── RGB/
├── T/
└── GT/
```

Images are matched by filename. The scripts currently use `RGB/`, `T/`, and `GT/` subdirectories for both training and testing data.

### Datasets

- TVCO1K: [Download](https://drive.google.com/file/d/1zFClM5N4pph71UzGwRMQX1OcXs6ctb_s/view?usp=drive_link)
- EVIAC: [Download](https://drive.google.com/file/d/1jGN2QihMTbbk5GK1eAJk7jo77qk0XPBO/view?usp=drive_link)

### Benchmark

- [Download the benchmark](https://drive.google.com/file/d/12pVGwz9lhpWOYo7bvIQ8YOkQQUNKhirT/view?usp=drive_link)

## Usage

The scripts use PyTorch and expect a CUDA-capable GPU. Install PyTorch and the other required Python packages in your environment before running them.

### Train

```bash
python train.py \
  --train_root /path/to/dataset/train \
  --val_root /path/to/dataset/test \
  --save_path ./checkpoints/dataset/ \
  --backbone resnet101d \
  --pretrained_path /path/to/pretrained_backbone.pth
```

Optional arguments include `--epoch`, `--lr`, `--batchsize`, `--trainsize`, `--resume`, and `--start_epoch`. Run `python train.py --help` to see the defaults and available options.

### Generate saliency maps

Download our ResNet-based model weights:

- [Download weights](https://drive.google.com/drive/folders/1t7vDYRccp6mnC-fBcqA6AFJTjaXa7eaZ?usp=drive_link)

```bash
python test.py \
  --test_dataset_path /path/to/dataset/test \
  --pth_path /path/to/checkpoint.pth \
  --save_path ./predictions/dataset/ \
  --backbone resnet101d
```

### Evaluate

`eval.py` computes S-measure, E-measure, F-measure, weighted F-measure, and MAE. Before running it, set the ground-truth directory and prediction directory in the script to match your local paths.

Our saliency maps are available here:

- [Download saliency maps](https://drive.google.com/drive/folders/12l9hv_T_cU1G0md_v0c4n8vwgbNaBtUJ?usp=drive_link)

## Citation

Citation information will be added when the associated publication details are available.

## License

License information will be added with the public release.
