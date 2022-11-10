from pathlib import Path

import numpy as np

exp_name = 'test_v2_yolox'

PROJECT_BASE = Path(r"D:\Projects\FishTracking")
DATA_BASE = Path(r"D:\Dataset\3DZeF20")

DETECTOR = 'd-detr'  # yolox | yolo4 | gt | d-detr
DETECTION_FILENAME = {
    'gt': 'gt_h',
    'yolo4': 'yolo4_h_zy',
    # 'yolox1': 'yolox_h_sc_1',
    # 'yolox': 'yolox_h_o21'
    # 'yolox': 'yolox_h_sc'
    'yolox': 'yolox_h_hc',
    'd-detr': 'ddetr_h'
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
# calc_seqs = [4]
calc_seqs = [1, 2, 3, 4]
# calc_seqs = [1, 2]
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
        'params': [40, 0, 0.5, 0.1]
        # 'params': [45, 0, 3.1, 0.3]
        # 'params': [40, 0, 0.5, 0.1]
    },
    "magic_numbers": {
        'min_err_thresh': 25,
        # ========================
        'max_err_thresh': 25,
        'max_app_thresh': 0.7,
        # 'min_err_thresh': 15,
        # 'max_err_thresh': 80,
        # 'max_app_thresh': 0.7,

# {'app_ratio_3d': 0.30000000000000004, 'cost_threshold': 3.1, 'max_age': 50.0, 'max_app_thresh': 0.30000000000000004, 'max_err_thresh': 60.0, 'min_err_thresh': 25.0}
# 02 MOTA:[ 91.60% ] IDF1:[ 81.30% ] precision:[ 96.30% ] recall:[ 95.30% ] fp:[165] fn:[212] ids:[3]
# 03 MOTA:[ 97.50% ] IDF1:[ 98.70% ] precision:[ 98.80% ] recall:[ 98.70% ] fp:[43] fn:[47] ids:[0]
# 04 MOTA:[ 86.70% ] IDF1:[ 90.50% ] precision:[ 94.20% ] recall:[ 92.50% ] fp:[258] fn:[343] ids:[2]
# {'metrics': 5.463, 'metrics_mota': 2.758, 'metrics_idf1': 2.705}
#
# {'app_ratio_3d': 0.30000000000000004, 'cost_threshold': 4.7, 'max_age': 45.0, 'max_app_thresh': 0.6000000000000001, 'max_err_thresh': 50.0, 'min_err_thresh': 55.0}
# 01 MOTA:[ 95.20% ] IDF1:[ 87.90% ] precision:[ 97.90% ] recall:[ 97.30% ] fp:[306] fn:[382] ids:[9]
# 02 MOTA:[ 90.80% ] IDF1:[ 80.90% ] precision:[ 95.90% ] recall:[ 94.90% ] fp:[182] fn:[229] ids:[5]
# 03 MOTA:[ 97.70% ] IDF1:[ 98.90% ] precision:[ 98.90% ] recall:[ 98.80% ] fp:[39] fn:[43] ids:[0]
# 04 MOTA:[ 85.20% ] IDF1:[ 89.50% ] precision:[ 93.50% ] recall:[ 91.70% ] fp:[290] fn:[376] ids:[6]
# {'metrics': 7.261, 'metrics_mota': 3.6889999999999996, 'metrics_idf1': 3.5720000000000005}

# {'app_ratio_3d': 0.1, 'cost_threshold': 4.7, 'max_age': 75.0, 'max_app_thresh': 0.7000000000000001, 'max_err_thresh': 25.0, 'min_err_thresh': 80.0, 'min_hits': 0.0}
# 01 MOTA:[ 95.70% ] IDF1:[ 97.90% ] precision:[ 97.90% ] recall:[ 97.80% ] fp:[304] fn:[311] ids:[2]
# 02 MOTA:[ 90.80% ] IDF1:[ 74.50% ] precision:[ 95.90% ] recall:[ 95.00% ] fp:[181] fn:[226] ids:[5]
# 03 MOTA:[ 97.50% ] IDF1:[ 98.70% ] precision:[ 98.80% ] recall:[ 98.70% ] fp:[44] fn:[47] ids:[0]
# 04 MOTA:[ 85.60% ] IDF1:[ 90.00% ] precision:[ 93.60% ] recall:[ 91.90% ] fp:[285] fn:[369] ids:[2]
# {'metrics': 7.307, 'metrics_mota': 3.6959999999999997, 'metrics_idf1': 3.611}
# {'p1': 75.0, 'p2': 0.0, 'p3': 1.3, 'p4': 50.0, 'p5': 60.0, 'p6': 0.30000000000000004, 'p7': 0.1}
    },
} for seq in SEQUENCES]

#
# ZebraFish-01 3D MOTA:[ 94.20% ] IDF1:[ 42.00% ] precision:[ 97.50% ] recall:[ 96.80% ] fp:[358] fn:[461] ids:[16]
# 2022-06-11 16:11:21.944442
# ZebraFish-02 3D MOTA:[ 89.20% ] IDF1:[ 78.10% ] precision:[ 95.70% ] recall:[ 93.60% ] fp:[190] fn:[289] ids:[5]
# 2022-06-11 16:11:21.946442
# ZebraFish-03 3D MOTA:[ 97.40% ] IDF1:[ 98.70% ] precision:[ 98.70% ] recall:[ 98.70% ] fp:[45] fn:[47] ids:[0]
# 2022-06-11 16:11:21.948483
# ZebraFish-04 3D MOTA:[ 87.20% ] IDF1:[ 86.80% ] precision:[ 94.30% ] recall:[ 92.90% ] fp:[256] fn:[324] ids:[3]