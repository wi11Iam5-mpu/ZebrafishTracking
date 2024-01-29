from pathlib import Path

import numpy as np

exp_name = 'test_v2_yolox'

PROJECT_BASE = Path(r"D:\Projects\FishTracking\for_release\ZebrafishTracking")
DATA_BASE = Path(r"D:\Dataset\3DZeF20")

DETECTOR = 'yolox_base'  # yolox | gt
DETECTION_FILENAME = {
    'gt': 'gt_h',
    'yolox_base': 'yolox_h_hc_base'
}

SEQUENCES = [
    {'index': '01', 'fish_number': 2, 'split': 'train'},
    {'index': '02', 'fish_number': 5, 'split': 'train'},
    {'index': '03', 'fish_number': 2, 'split': 'train'},
    {'index': '04', 'fish_number': 5, 'split': 'train'},
    {'index': '05', 'fish_number': 1, 'split': 'test'},
    {'index': '06', 'fish_number': 2, 'split': 'test'},
    {'index': '07', 'fish_number': 5, 'split': 'test'},
    {'index': '08', 'fish_number': 10, 'split': 'test'}
]

train = True  # False True
calc_seqs = [1, 2, 3, 4]  # 1, 2, 3, 4 Index of Seqs
calc_seqs = [s - 1 for s in calc_seqs]
SEQUENCES = np.array(SEQUENCES)[calc_seqs].tolist() if train else SEQUENCES[4:]

configs = [{
    "index": seq['index'],
    "fish_number": seq['fish_number'],
    "embedding_model_path":
        PROJECT_BASE / "modules" / "deep" / "weight" / "exp_res_4data_03_cdc_resnet18_128_mlp_64_bs128_checkpoint_223.pth",
    "detection_path":
        PROJECT_BASE / "sequences" / "detections" / DETECTION_FILENAME[DETECTOR] / '2d_detections' / seq['index'],
    "image_path": DATA_BASE / seq['split'] / f"ZebraFish-{seq['index']}",

    "calibration_path":
        PROJECT_BASE / "sequences" / 'calibrations' / seq['index'],
    "gt_path":
        PROJECT_BASE / "sequences" / "gt_files" / seq['index'] / "gt.txt",
    "output_path":
        PROJECT_BASE / "sequences" / 'outputs' / seq['index'],
    "3d_tracker": {
        # max_age, cost_threshold, app_ratio_3d
        'params': [77, 0, 2.5, 0.3]
    },
    "magic_numbers": {
        'min_err_thresh': 23,
        # ========================
        'max_err_thresh': 55,
        'max_app_thresh': 0.7,
    },
    # "detector": DETECTOR
} for seq in SEQUENCES]
