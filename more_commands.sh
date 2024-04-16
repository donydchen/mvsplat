#!/usr/bin/env bash

# In this file, we provide commands to get the quantitative results presented in the MVSplat paper.
# Commands are provided by following the order of Tables appearing in the paper.


# --------------- Default Final Models ---------------

# Table 1: re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true

# Table 1: acid
python -m src.main +experiment=acid \
checkpointing.load=checkpoints/acid.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true

# generate video
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=false


# --------------- Cross-Dataset Generalization ---------------

# Table 2: RealEstate10K -> ACID
python -m src.main +experiment=acid \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true

# Table 2: RealEstate10K -> DTU (2 context views)
python -m src.main +experiment=dtu \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx2.json \
test.compute_scores=true

# RealEstate10K -> DTU (3 context views)
python -m src.main +experiment=dtu \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_dtu_nctx3.json \
dataset.view_sampler.num_context_views=3 \
wandb.name=dtu/views3 \
test.compute_scores=true


# --------------- Ablation Models ---------------

# Table 3: base
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_base \
model.encoder.wo_depth_refine=true 

# Table 3: w/o cost volume
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wocv.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wocv \
model.encoder.wo_depth_refine=true \
model.encoder.wo_cost_volume=true

# Table 3: w/o cross-view attention
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wobbcrossattn_best.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_backbone_cross_attn \
model.encoder.wo_depth_refine=true \
model.encoder.wo_backbone_cross_attn=true

# Table 3: w/o U-Net
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wounet.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_unet \
model.encoder.wo_depth_refine=true \
model.encoder.wo_cost_volume_refine=true

# Table B: w/ Epipolar Transformer
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wepitrans.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_w_epipolar_trans \
model.encoder.wo_depth_refine=true \
model.encoder.use_epipolar_trans=true

# Table C: 3 Gaussians per pixel
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_gpp3.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_gpp3 \
model.encoder.gaussians_per_pixel=3

# Table D: w/ random init (300K)
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_wopretrained.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_pretrained 

# Table D: w/ random init (450K)
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_wopretrained_450k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_pretrained_450k 
