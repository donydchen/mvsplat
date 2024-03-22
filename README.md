# MVSplat

This is the official implementation of **MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images** by Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai.

### [Project Page](https://donydchen.github.io/mvsplat/) | [arXiv](https://arxiv.org/abs/2403.14627) | [Pretrained Models](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU) 

https://github.com/donydchen/mvsplat/assets/5866866/c5dc5de1-819e-462f-85a2-815e239d8ff2

## Installation

To get started, create a conda virtual environment using Python 3.10+ and install requirements:

```bash
conda create -n mvsplat python=3.10
conda activate mvsplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Acquiring Datasets

MVSplat utilises the same dataset settings as pixelSplat. Below we quote pixelSplat's [detailed instructions](https://github.com/dcharatan/pixelsplat?tab=readme-ov-file#acquiring-datasets) on getting datasets.

> pixelSplat was trained using versions of the RealEstate10k and ACID datasets that were split into ~100 MB chunks for use on server cluster file systems. Small subsets of the Real Estate 10k and ACID datasets in this format can be found [here](https://drive.google.com/drive/folders/1joiezNCyQK2BvWMnfwHJpm2V77c7iYGe?usp=sharing). To use them, simply unzip them into a newly created `datasets` folder in the project root directory.

> If you would like to convert downloaded versions of the Real Estate 10k and ACID datasets to our format, you can use the [scripts here](https://github.com/dcharatan/real_estate_10k_tools). Reach out to us (pixelSplat) if you want the full versions of our processed datasets, which are about 500 GB and 160 GB for Real Estate 10k and ACID respectively.

## Run the Code

### Evaluation

To render frames and compute scores from an existing checkpoint,

* get the [pretrained models](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU), and save them to `/checkpoints`

* run the following:

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true

# acid
python -m src.main +experiment=acid \
checkpointing.load=checkpoints/acid.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true
```

* the rendered novel views will be stored under `outputs/test`

To render videos from a pretrained model, run the following

```bash
# re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=false
```

### Training

Run the following:

```bash
# download the backbone pretrained weight from unimath and save to 'checkpoints/'
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
# train mvsplat
python -m src.main +experiment=re10k data_loader.train.batch_size=14
```

The setting requires a single GPU with 80 GB of VRAM (A100). Set a smaller `data_loader.train.batch_size` to reduce memory usage.

### Ablations

We also provide a collection of our [ablation models](https://drive.google.com/drive/folders/14_E_5R6ojOWnLSrSVLVEMHnTiKsfddjU) (under folder 'ablations'). To evaluate them, *e.g.*, the 'base' model, run the following command

```bash
# Table 3: base
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_base \
model.encoder.wo_depth_refine=true 
```

More running commands can be found at [more_commands.sh](more_commands.sh).

## BibTeX

```bibtex
@article{chen2024mvsplat,
    title   = {MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images},
    author  = {Chen, Yuedong and Xu, Haofei and Zheng, Chuanxia and Zhuang, Bohan and Pollefeys, Marc and Geiger, Andreas and Cham, Tat-Jen and Cai, Jianfei},
    journal = {arXiv preprint arXiv:2403.14627},
    year    = {2024},
}
```

## Acknowledgements

The project is largely based on [pixelSplat](https://github.com/dcharatan/pixelsplat) and has incorporated numerous code snippets from [UniMatch](https://github.com/autonomousvision/unimatch). Many thanks to these two projects for their excellent contributions!
