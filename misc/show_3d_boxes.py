import itertools
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from packaging import version

from modules.trackers.mht.mht import MHTTracker
from modules.trackers.sort.sort_3d_weighted import Detection3D, Sort3DWeighted
from modules.trackers.tracker_base import ConstructFirstMethod

if version.parse(sklearn.__version__) > version.parse("0.22"):
    import joblib
else:
    pass

from modules.reconstruction.Triangulate import Triangulate
from modules.utils.misc import time_count
import numpy as np
from collections import defaultdict

"""
    计算2D_bbox中心点，计算3D长宽高（选大的一边），默认身体一直是树立的。
"""


def point3d_construct(point1, point2):
    tr = Triangulate()
    p, d = tr.triangulatePoint(point1,
                               point2,
                               cam1_obj,
                               cam2_obj,
                               correctRefraction=True)

    p1 = cam1_obj.forwardprojectPoint(*p)
    p2 = cam2_obj.forwardprojectPoint(*p)

    return p, p1, p2

def plot_boxes(df_box, ax):
    #
    top_p1 = (df_box['camT_left'].values[0], df_box['camT_top'].values[0])
    top_p2 = (df_box['camT_left'].values[0] + df_box['camT_width'].values[0], df_box['camT_top'].values[0])
    top_p3 = (df_box['camT_left'].values[0], df_box['camT_top'].values[0] + df_box['camT_height'].values[0])
    top_p4 = (df_box['camT_left'].values[0] + df_box['camT_width'].values[0],
              df_box['camT_top'].values[0] + df_box['camT_height'].values[0])
    #
    front_p1 = (df_box['camF_left'].values[0], df_box['camF_top'].values[0])
    front_p2 = (df_box['camF_left'].values[0] + df_box['camF_width'].values[0], df_box['camF_top'].values[0])
    front_p3 = (df_box['camF_left'].values[0], df_box['camF_top'].values[0] + df_box['camF_height'].values[0])
    front_p4 = (df_box['camF_left'].values[0] + df_box['camF_width'].values[0],
                df_box['camF_top'].values[0] + df_box['camF_height'].values[0])
    #
    top_p1 = (float(top_p1[0]), float(top_p1[1]))
    top_p2 = (float(top_p2[0]), float(top_p2[1]))
    top_p3 = (float(top_p3[0]), float(top_p3[1]))
    top_p4 = (float(top_p4[0]), float(top_p4[1]))
    #
    front_p1 = (float(front_p1[0]), float(front_p1[1]))
    front_p2 = (float(front_p2[0]), float(front_p2[1]))
    front_p3 = (float(front_p3[0]), float(front_p3[1]))
    front_p4 = (float(front_p4[0]), float(front_p4[1]))
    # p1  p2   top
    # p3  p4
    # p1  p2   front
    # p3  p4
    p11 = point3d_construct(top_p1, front_p1)
    p13 = point3d_construct(top_p1, front_p3)
    p22 = point3d_construct(top_p2, front_p2)
    p24 = point3d_construct(top_p2, front_p4)
    p33 = point3d_construct(top_p3, front_p3)
    p31 = point3d_construct(top_p3, front_p1)
    p44 = point3d_construct(top_p4, front_p4)
    p42 = point3d_construct(top_p4, front_p2)
    points = [p11, p13, p22, p24, p33, p31, p44, p42]
    labels = ['p11', 'p13', 'p22', 'p24', 'p33', 'p31', 'p44', 'p42']
    # points = [p13, p24, p31, p42]
    # labels = ['p13', 'p24', 'p31', 'p42']
    x_3d, y_3d, z_3d = np.array([]), np.array([]), np.array([])
    x_2d_top, y_2d_top, z_2d_top = np.array([]), np.array([]), np.array([])
    x_2d_front, y_2d_front, z_2d_front = np.array([]), np.array([]), np.array([])
    for point, label in zip(points, labels):
        pt = point[0]
        pt_top = point[1]
        pt_front = point[2]
        x_3d = np.append(x_3d, pt[0])
        y_3d = np.append(y_3d, pt[1])
        z_3d = np.append(z_3d, pt[2])
        # ax.text(pt[0], pt[1], pt[2], f"{label}{np.array(pt, dtype=int)}")
        # ax.text(pt[0], pt[1], pt[2], f"{label}")

        x_2d_top = np.append(x_2d_top, pt_top[0] / factor)
        y_2d_top = np.append(y_2d_top, pt_top[1] / factor)
        z_2d_top = np.append(z_2d_top, 30)

        x_2d_front = np.append(x_2d_front, pt_front[0] / factor)
        y_2d_front = np.append(y_2d_front, 0)
        z_2d_front = np.append(z_2d_front, pt_front[1] / factor)
    ax.scatter(x_3d, y_3d, z_3d, c='r', s=5)
    ax.scatter(x_2d_top, y_2d_top, z_2d_top, c='g', s=5)
    ax.scatter(x_2d_front, y_2d_front, z_2d_front, c='b', s=5)
    selected_corners = [p13[1], p31[1], p42[1], p24[1]]
    prev = selected_corners[-1]
    for corner in selected_corners:
        ax.plot([prev[0] / factor, corner[0] / factor], [prev[1] / factor, corner[1] / factor], [30, 30],
                color='g',
                linewidth=1)
        prev = corner
    selected_corners = [p13[2], p31[2], p42[2], p24[2]]
    prev = selected_corners[-1]
    for corner in selected_corners:
        ax.plot([prev[0] / factor, corner[0] / factor], [0, 0], [prev[1] / factor, corner[1] / factor],
                color='b',
                linewidth=1)
        prev = corner

    selected_corners = [p13[0], p24[0], p22[0], p11[0]]
    prev = selected_corners[-1]
    for corner in selected_corners:
        ax.plot([prev[0], corner[0]],
                [prev[1], corner[1]],
                [prev[2], corner[2]],
                color='r',
                linewidth=1)
        prev = corner

    selected_corners = [p33[0], p44[0], p42[0], p31[0]]
    prev = selected_corners[-1]
    for corner in selected_corners:
        ax.plot([prev[0], corner[0]],
                [prev[1], corner[1]],
                [prev[2], corner[2]],
                color='r',
                linewidth=1)
        prev = corner

    for prev, corner in zip([p13[0], p24[0], p22[0], p11[0]], [p33[0], p44[0], p42[0], p31[0]]):
        ax.plot([prev[0], corner[0]],
                [prev[1], corner[1]],
                [prev[2], corner[2]],
                color='r',
                linewidth=1)

factor = 100
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection="3d")

# Setting the axes properties
# 2704 * 1520
ax1.set_xlim3d([0, 30])
ax1.set_xlabel('X')
ax1.set_ylim3d([0, 30])
ax1.set_ylabel('Y')
ax1.set_zlim3d([0, 30])
ax1.set_zlabel('Z')

# Provide starting angle for the view.
ax1.view_init(25, 45)

index = '02'
frame = 99

calibration_path = Path(fr"D:\Projects\FishTracking\sequences\calibrations\{index}")
# Reading calibration weight
cam1_obj = calibration_path / 'cam1.pkl'
cam2_obj = calibration_path / 'cam2.pkl'

cam1_obj = joblib.load(cam1_obj)
cam1_obj.calcExtrinsicFromJson(calibration_path / 'cam1_references.json')

cam2_obj = joblib.load(cam2_obj)
cam2_obj.calcExtrinsicFromJson(calibration_path / 'cam2_references.json')

gt_file = Path(fr"D:\Projects\FishTracking\sequences\gt_files\{index}\gt.txt")

# ['frame', 'id', '3d_x', '3d_y', '3d_z', 'camT_x', 'camT_y', 'camT_left',
#        'camT_top', 'camT_width', 'camT_height', 'camT_occlusion', 'camF_x',
#        'camF_y', 'camF_left', 'camF_top', 'camF_width', 'camF_height',
#        'camF_occlusion']
gt_df = pd.read_csv(gt_file)
gt_df = gt_df[
    ['frame', 'id', 'camT_left', 'camT_top', 'camT_width', 'camT_height', 'camF_left', 'camF_top', 'camF_width',
     'camF_height']]

gt_df = gt_df[gt_df["frame"] == frame]
ids = gt_df['id'].unique()

for _id in ids:
    df_box = gt_df[gt_df["id"] == _id]
    plot_boxes(df_box, ax1)

plt.show()
