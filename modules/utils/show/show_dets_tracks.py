from collections import deque, defaultdict
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def put_mask(frame, bbox1):
    def rotate(img, angle, center):
        rows, cols = img.shape[:2]

        M = cv2.getRotationMatrix2D(center, angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    # 1.读取图片
    image = frame
    # 2.获取标签
    # 标签格式　bbox = [xl, yl, xr, yr]
    # bbox1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    # center1 = ((25 + bbox1[0]) // 2, (25 + bbox1[1]) // 2)
    center1 = (bbox1[0], bbox1[1])
    # 3.画出mask
    zeros1 = np.zeros(image.shape, dtype=np.uint8)
    zeros_mask1 = cv2.rectangle(zeros1,
                                (int(bbox1[0] - 25), int(bbox1[1]) - 25),
                                (int(bbox1[0] + 25), int(bbox1[1] + 25)),
                                color=(0, 255, 0), thickness=-1)  # thickness=-1 表示矩形框内颜色填充
    # zeros_mask1 = cv2.circle(zeros1, (int(center1[0]), int(center1[1])), 10, color=(0, 255, 0),
    #                          thickness=-1)  # thickness=-1 表示矩形框内颜色填充

    zeros_mask = np.array(zeros_mask1)
    # zeros_mask = rotate(zeros_mask, 0, (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2)))
    # alpha 为第一张图片的透明度
    alpha = 1
    # beta 为第二张图片的透明度
    beta = 0.1
    gamma = 0
    # cv2.addWeighted 将原始图片与 mask 融合
    mask_img = cv2.addWeighted(image, alpha, zeros_mask, beta, gamma)

    return mask_img


# D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\detections\yolo4_h_zy\2d_detections\01

detections = 'yolox_h_hc'  # 'yolo4_h_zy'


def main(seq_index='01'):
    videos = ['cam1', 'cam2']
    input_path = Path(
        r"D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences")

    raw_video1 = input_path / 'videos' / seq_index / 'cam1.mp4'
    raw_video2 = input_path / 'videos' / seq_index / 'cam2.mp4'

    show_det1 = input_path / 'detections' / detections / '2d_detections' / seq_index / 'detections_2d_cam1.csv'
    show_det2 = input_path / 'detections' / detections / '2d_detections' / seq_index / 'detections_2d_cam2.csv'

    IS_SAVE = True

    title = '2d_3d'
    fps = 60
    code = 'MP4V'
    output_video_name = f'./det_{seq_index}.mp4'
    det_path = [show_det1, show_det2]
    raw_video_path = [raw_video1, raw_video2]
    data_pack = [pd.read_csv(p, header=None) for p in det_path]
    data_pack[0].columns = ['frame', 'x', 'y', 'c']
    data_pack[1].columns = ['frame', 'x', 'y', 'c']
    data_pack[0] = data_pack[0][['frame', 'x', 'y']]
    data_pack[1] = data_pack[1][['frame', 'x', 'y']]
    selected = ['frame', 'x', 'y']
    data_pack_selected = [data[selected].values for i, data in enumerate(data_pack)]

    readers = [cv2.VideoCapture(str(p)) for p in raw_video_path]

    frame_count = int(readers[0].get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(readers[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(readers[0].get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*code)
    writer = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_width, 2 * frame_height))

    s_factor = 3
    set_frame_width = int(frame_width // s_factor)
    set_frame_height = int(frame_height // s_factor)

    bbox = [defaultdict(list), defaultdict(list)]

    for i, data in enumerate(data_pack_selected):
        for d in data:
            bbox[i][int(d[0])].append(d[1:])

    show_trk = input_path / 'outputs' / seq_index / 'all_dets.csv'
    title = 'dets & tracks'

    data_pack = pd.read_csv(show_trk)
    ids = list(set(data_pack['id'].values))

    tracks_selected = [['frame', 'id', 't_x', 't_y'],
                       ['frame', 'id', 'f_x', 'f_y']]

    tracks_data_pack_selected = [data_pack[labels].values for labels in tracks_selected]

    clrs_n = len(ids) + 1000
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for _ in range(clrs_n)]

    tracks_bbox = [defaultdict(list), defaultdict(list)]

    for i, data in enumerate(tracks_data_pack_selected):
        for d in data:
            tracks_bbox[i][int(d[0])].append(d[1:])

    while readers[0].isOpened() and readers[1].isOpened():
        max_len = 200
        q = deque(maxlen=max_len)
        for n_frame in range(1, frame_count + 1):
            ret, frame1 = readers[0].read()
            ret, frame2 = readers[1].read()
            cv2.namedWindow(title, cv2.WINDOW_FREERATIO)
            cv2.resizeWindow(title, set_frame_width // 1, 2 * set_frame_height // 1)

            cv2.putText(frame1, f'frame: {n_frame}', (100, 150), cv2.FONT_HERSHEY_PLAIN, 5, color=(0, 0, 255),
                        thickness=3)

            if n_frame in bbox[0].keys():
                for i, bb in enumerate(bbox[0][n_frame]):
                    frame1 = put_mask(frame1, bb)

            if n_frame in bbox[1].keys():
                for i, bb in enumerate(bbox[1][n_frame]):
                    frame2 = put_mask(frame2, bb)

            if n_frame in tracks_bbox[0].keys():
                for i, bb in enumerate(tracks_bbox[0][n_frame]):
                    cv2.circle(frame1, (int(bb[1]), int(bb[2])), 10, color=colors[int(bb[0])], thickness=-1)
                    cv2.putText(frame1, str(int(bb[0])), (int(bb[1]), int(bb[2])), cv2.FONT_HERSHEY_PLAIN, 3,
                                color=colors[int(bb[0])],
                                thickness=3)

            if n_frame in tracks_bbox[1].keys():
                for i, bb in enumerate(tracks_bbox[1][n_frame]):
                    cv2.circle(frame2, (int(bb[1]), int(bb[2])), 10, color=colors[int(bb[0])], thickness=-1)
                    cv2.putText(frame2, str(int(bb[0])), (int(bb[1]), int(bb[2])), cv2.FONT_HERSHEY_PLAIN, 3,
                                color=colors[int(bb[0])],
                                thickness=3)

            frame = np.vstack((frame1, frame2))

            cv2.imshow(title, frame)
            frame = cv2.resize(frame, (frame_width, 2 * frame_height))
            if IS_SAVE: writer.write(frame)

            # cv2.waitKey(0)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                cv2.waitKey(0)
            if key == ord('q'):
                break
        break

    readers[0].release()
    readers[1].release()
    cv2.destroyAllWindows()
    writer.release()
    writer.release()


if __name__ == '__main__':
    # for i in ['01', '02', '03', '04', '05', '06', '07', '08']:
    for i in ['03']:
        main(i)
