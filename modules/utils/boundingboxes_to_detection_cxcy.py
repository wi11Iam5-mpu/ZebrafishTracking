import os
from pathlib import Path

import numpy as np
import pandas as pd

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

cols = ['Filename', 'Object ID', 'Annotation tag',
        'Upper left corner X',
        'Upper left corner Y',
        'Lower right corner X',
        'Lower right corner Y', 'Confidence', 'Frame']

# detection = 'yolox_h_sc'
detection = 'yolox_h_hc'


def main(index, cam='cam1'):
    output = Path(
        fr'D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\detections\{detection}\2d_detections\{index}\detections_2d_{cam}.csv'
    )

    xy_file = Path(
        fr'D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\detections\{detection}\2d_detections\{index}\boundingboxes_2d_{cam}.csv'
    )

    xy_df = pd.read_table(xy_file, sep=',')
    # xy_df.columns = cols

    xy_df = xy_df[xy_df["Confidence"] >= 0.1]

    bbox_lx = xy_df[['Upper left corner X']].values
    bbox_ly = xy_df[['Upper left corner Y']].values
    bbox_rx = xy_df[['Lower right corner X']].values
    bbox_ry = xy_df[['Lower right corner Y']].values
    bbox_cx = ((bbox_rx + bbox_lx) / 2).astype(float)
    bbox_cy = ((bbox_ry + bbox_ly) / 2).astype(float)
    bbox_frame = xy_df[['Frame']].values.astype(float)
    bbox_conf = xy_df[['Confidence']].values.astype(float)

    df_final = pd.DataFrame(np.hstack([bbox_frame, bbox_cx, bbox_cy, bbox_conf]))

    print(df_final)
    df_final.to_csv(output, index=False, header=False)


# python BgDetector.py -f D:\MOT_Research\MOT3D\FishTracking\sequences\01\detection\gt\ -c 1 -pd
# python BgDetector.py -f D:\MOT_Research\MOT3D\FishTracking\sequences\01\detection\gt\ -c 2 -pd

if __name__ == '__main__':
    for cam in ["cam1", "cam2"]:
        for seq in ['01', '02', '03', '04', '05', '06', '07', '08']:
            main(seq, cam)
