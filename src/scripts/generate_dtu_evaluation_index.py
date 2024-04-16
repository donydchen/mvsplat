''' Build upon: https://github.com/autonomousvision/murf/blob/main/datasets/dtu.py 
'''

import torch
import os
from glob import glob
import argparse
from einops import rearrange, repeat
from dataclasses import asdict, dataclass
import json
from tqdm import tqdm
import numpy as np


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]


def sorted_test_src_views_fixed(cam2worlds_dict, test_views, train_views):
    # use fixed src views for testing, instead of for using different src views for different test views
    cam_pos_trains = np.stack([cam2worlds_dict[x] for x in train_views])[
        :, :3, 3
    ]  # [V, 3], V src views
    cam_pos_target = np.stack([cam2worlds_dict[x] for x in test_views])[
        :, :3, 3
    ]  # [N, 3], N test views in total
    dis = np.sum(
        np.abs(cam_pos_trains[:, None] - cam_pos_target[None]), axis=(1, 2))
    src_idx = np.argsort(dis)
    src_idx = [train_views[x] for x in src_idx]

    return src_idx


def main(args):
    data_dir = os.path.join("datasets", args.dataset_name, "test")

    # load view pairs
    # Adopt from: https://github.com/autonomousvision/murf/blob/main/datasets/dtu.py#L95
    test_views = [32, 24, 23, 44]
    train_views = [i for i in range(49) if i not in test_views]

    index = {}
    for torch_file in tqdm(sorted(glob(os.path.join(data_dir, "*.torch")))):
        scene_datas = torch.load(torch_file)
        for scene_data in scene_datas:
            cameras = scene_data["cameras"]
            scene_name = scene_data["key"]

            # calculate nearest camera index
            w2c = repeat(
                torch.eye(4, dtype=torch.float32),
                "h w -> b h w",
                b=cameras.shape[0],
            ).clone()
            w2c[:, :3] = rearrange(
                cameras[:, 6:], "b (h w) -> b h w", h=3, w=4)
            opencv_c2ws = w2c.inverse()  # .unsqueeze(0)
            xyzs = opencv_c2ws[:, :3, -1].unsqueeze(0)  # 1, N, 3
            cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
            cameras_dist_index = torch.argsort(
                cameras_dist_matrix, dim=-1).squeeze(0)

            cam2worlds_dict = {k: v for k, v in enumerate(opencv_c2ws)}
            nearest_fixed_views = sorted_test_src_views_fixed(
                cam2worlds_dict, test_views, train_views
            )

            selected_pts = test_views
            for seq_idx, cur_mid in enumerate(selected_pts):
                cur_nn_index = nearest_fixed_views
                contexts = tuple([int(x)
                                 for x in cur_nn_index[: args.n_contexts]])
                targets = (cur_mid,)
                index[f"{scene_name}_{seq_idx:02d}"] = IndexEntry(
                    context=contexts,
                    target=targets,
                )
    # save index to files
    out_path = f"assets/evaluation_index_{args.dataset_name}_nctx{args.n_contexts}.json"
    with open(out_path, "w") as f:
        json.dump({k: None if v is None else asdict(v)
                  for k, v in index.items()}, f)
    print(f"Dumped index to: {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_contexts", type=int,
                        default=2, help="output directory")
    parser.add_argument("--dataset_name", type=str, default="dtu")

    params = parser.parse_args()

    main(params)
