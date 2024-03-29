import numpy as np


# From https://github.com/open-mmlab/mmdetection3d/blob/fcb4545ce719ac121348cab59bac9b69dd1b1b59/mmdet3d/datasets/scannet_dataset.py
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.
    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, seg_label):
        """Call function to map original semantic class to valid category ids.
        Args:
            results (dict): Result dict containing point semantic masks.
        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.
                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        seg_label = np.clip(seg_label, 0, self.max_cat_id)
        return self.cat_id2class[seg_label]