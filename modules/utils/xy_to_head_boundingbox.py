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


def main(index, cam='cam1'):
    output = Path(
        fr'D:\Datasets\3DZeF20\test\ZebraFish-{index}\processed\boundingboxes_2d_{cam}.csv'
    )

    xy_file = Path(
        fr'D:\Projects\FishTracking\sequences\detections\yolo4_h_zy\2d_detections\{index}\detections_2d_{cam}.csv'
    )

    side = 0
    if cam == 'cam1':
        side = 26
    else:
        side = 50

    xy_df = pd.read_table(xy_file, sep=',', header=None)
    xy_df.columns = ['frame', 'x', 'y']
    frame_df = xy_df[['frame']]
    frame_df.columns = ['Frame']
    frame_df.reset_index(drop=True, inplace=True)
    # static diameter 25
    bbox_lx = xy_df[['x']] - side / 2
    bbox_lx.reset_index(drop=True, inplace=True)
    bbox_ly = xy_df[['y']] - side / 2
    bbox_ly.reset_index(drop=True, inplace=True)
    bbox_rx = xy_df[['x']] + side / 2
    bbox_rx.reset_index(drop=True, inplace=True)
    bbox_ry = xy_df[['y']] + side / 2
    bbox_ry.reset_index(drop=True, inplace=True)
    bbox = pd.concat([bbox_lx, bbox_ly, bbox_rx, bbox_ry], axis=1)
    bbox.columns = ['Upper left corner X',
                    'Upper left corner Y',
                    'Lower right corner X',
                    'Lower right corner Y']

    df_tmp = pd.concat([frame_df,
                        pd.DataFrame(columns=['Object ID', 'Annotation tag', 'Filename', 'Confidence'])],
                       axis=1)
    df_tmp.loc[:, 'Object ID'] = -1
    df_tmp.loc[:, 'Annotation tag'] = 'zebrafish'
    df_tmp.loc[:, 'Filename'] = [f"{int(fr):06}.png" for fr in frame_df.values]
    df_tmp.loc[:, 'Confidence'] = 1
    df_tmp.reset_index(drop=True, inplace=True)
    df_final = pd.concat([df_tmp, bbox], axis=1)

    df_final = df_final[cols]
    print(df_final)
    df_final.to_csv(output, index=None)


# python BgDetector.py -f D:\MOT_Research\MOT3D\FishTracking\sequences\01\detection\gt\ -c 1 -pd
# python BgDetector.py -f D:\MOT_Research\MOT3D\FishTracking\sequences\01\detection\gt\ -c 2 -pd

if __name__ == '__main__':
    for cam in ["cam1", "cam2"]:
        # for seq in ['01', '02', '03', '04']:
        for seq in ['05', '06', '07', '08']:
            main(seq, cam)
