import torch
import torch.utils.data
import numpy as np
import torchvision.utils as vutils
import cv2
from matplotlib.cm import get_cmap
import matplotlib as mpl
import matplotlib.cm as cm


# https://github.com/autonomousvision/unimatch/blob/master/utils/visualization.py


def vis_disparity(disp):
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    return disp_vis


def viz_depth_tensor(disp, return_numpy=False, colormap='plasma'):
    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)

    disp = disp.numpy()
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3]

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]

    return viz
