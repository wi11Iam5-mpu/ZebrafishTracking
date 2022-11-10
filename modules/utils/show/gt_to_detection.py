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


def main(index):
    output1 = Path(
        fr'D:\Datasets\3DZeF20\train\ZebraFish-{index}\processed\boundingboxes_2d_cam1.csv'
        # fr"D:\MOT_Research\MOT3D\FishTracking\sequences\{index}\detection\gt\processed\boundingboxes_2d_cam1.csv"
    )
    output2 = Path(
        fr'D:\Datasets\3DZeF20\train\ZebraFish-{index}\processed\boundingboxes_2d_cam2.csv'
        # fr"D:\MOT_Research\MOT3D\FishTracking\sequences\{index}\detection\gt\processed\boundingboxes_2d_cam2.csv"
    )
    gt_file = Path(fr"D:\Projects\FishTracking\sequences\{index}\gt\gt.txt")
    gt_df = pd.read_table(gt_file, sep=',')
    frame_df = gt_df[['frame']]
    frame_df.columns = ['Frame']
    frame_df.reset_index(drop=True, inplace=True)
    gt_df.loc[:, 'camT_width'] = gt_df.loc[:, 'camT_width'].values + gt_df.loc[:, 'camT_left'].values
    gt_df.loc[:, 'camT_height'] = gt_df.loc[:, 'camT_height'].values + gt_df.loc[:, 'camT_top'].values
    bbox_lx_cam1 = gt_df[['camT_left']]
    bbox_lx_cam1.reset_index(drop=True, inplace=True)
    bbox_ly_cam1 = gt_df[['camT_top']]
    bbox_ly_cam1.reset_index(drop=True, inplace=True)
    bbox_rx_cam1 = gt_df[['camT_width']]
    bbox_rx_cam1.reset_index(drop=True, inplace=True)
    bbox_ry_cam1 = gt_df[['camT_height']]
    bbox_ry_cam1.reset_index(drop=True, inplace=True)
    bbox_cam1 = pd.concat([bbox_lx_cam1, bbox_ly_cam1, bbox_rx_cam1, bbox_ry_cam1], axis=1)
    bbox_cam1.columns = ['Upper left corner X',
                         'Upper left corner Y',
                         'Lower right corner X',
                         'Lower right corner Y']
    gt_df.loc[:, 'camF_width'] = gt_df.loc[:, 'camF_width'].values + gt_df.loc[:, 'camF_left'].values
    gt_df.loc[:, 'camF_height'] = gt_df.loc[:, 'camF_height'].values + gt_df.loc[:, 'camF_top'].values
    bbox_lx_cam2 = gt_df[['camF_left']]
    bbox_lx_cam2.reset_index(drop=True, inplace=True)
    bbox_ly_cam2 = gt_df[['camF_top']]
    bbox_ly_cam2.reset_index(drop=True, inplace=True)
    bbox_rx_cam2 = gt_df[['camF_width']]
    bbox_rx_cam2.reset_index(drop=True, inplace=True)
    bbox_ry_cam2 = gt_df[['camF_height']]
    bbox_ry_cam2.reset_index(drop=True, inplace=True)
    bbox_cam2 = pd.concat([bbox_lx_cam2, bbox_ly_cam2, bbox_rx_cam2, bbox_ry_cam2], axis=1)
    bbox_cam2.columns = ['Upper left corner X',
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
    df_final_cam1 = pd.concat([df_tmp, bbox_cam1], axis=1)
    df_final_cam2 = pd.concat([df_tmp, bbox_cam2], axis=1)
    print(df_final_cam1.columns)
    df_final_cam1 = df_final_cam1[cols]
    df_final_cam2 = df_final_cam2[cols]
    df_final_cam1.to_csv(output1, index=None)
    df_final_cam2.to_csv(output2, index=None)


# python BgDetector.py -f D:\MOT_Research\MOT3D\FishTracking\sequences\01\detection\gt\ -c 1 -pd
# python BgDetector.py -f D:\MOT_Research\MOT3D\FishTracking\sequences\01\detection\gt\ -c 2 -pd

if __name__ == '__main__':
    for seq in ['01', '02', '03', '04']:
        main(seq)
