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
    center1 = (bbox1[0] + 13, bbox1[1] + 13)
    # 3.画出mask
    zeros1 = np.zeros(image.shape, dtype=np.uint8)
    zeros_mask1 = cv2.rectangle(zeros1,
                                (int(bbox1[0] - 25), int(bbox1[1]) - 25),
                                (int(bbox1[0] + 50), int(bbox1[1] + 50)),
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


# D:\MOT_Research\MOT3D\Updating\FishTracking\sequences\detections\yolo4_h_zy\2d_detections\01

detections = 'yolox_h_sc'  # 'yolo4_h_zy'
# detections = 'yolo4_h_zy'  # 'yolo4_h_zy'


def main(seq_index='01'):
    videos = ['cam1', 'cam2']
    input_path = Path(
        r"/sequences")

    raw_video1 = input_path / 'videos' / seq_index / 'cam1.mp4'
    raw_video2 = input_path / 'videos' / seq_index / 'cam2.mp4'

    show_det1 = input_path / 'detections' / detections / '2d_detections' / seq_index / 'boundingboxes_2d_cam1.csv'
    show_det2 = input_path / 'detections' / detections / '2d_detections' / seq_index / 'boundingboxes_2d_cam2.csv'

    IS_SAVE = True

    title = '2d'
    fps = 60
    code = 'MP4V'
    output_video_name = f'./det_{seq_index}.mp4'
    track_path = [show_det1, show_det2]
    raw_video_path = [raw_video1, raw_video2]
    data_pack = [pd.read_csv(p) for p in track_path]

    # cols = ['Filename', 'Object ID', 'Annotation tag',
    #         'Upper left corner X',
    #         'Upper left corner Y',
    #         'Lower right corner X',
    #         'Lower right corner Y', 'Confidence', 'Frame']

    data_pack[0] = data_pack[0][['Frame', 'Upper left corner X', 'Upper left corner Y']].values
    data_pack[1] = data_pack[1][['Frame', 'Upper left corner X', 'Upper left corner Y']].values

    # data_pack[0].columns = ['frame', 'cx', 'cy']
    # data_pack[1].columns = ['frame', 'cx', 'cy']
    # selected = ['frame', 'cx', 'cy']
    # data_pack_selected = [data[selected].values for i, data in enumerate(data_pack)]
    data_pack_selected = data_pack

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

            # related = defaultdict(list)
            if n_frame in bbox[0].keys():
                for i, bb in enumerate(bbox[0][n_frame]):
                    cv2.rectangle(frame1, pt1=(int(bb[0]), int(bb[1])),
                                  pt2=(int(bb[0]) + 26,
                                       int(bb[1]) + 26),
                                  color=(225, 225, 225), thickness=10)
                    # frame1 = put_mask(frame1, bb)

            if n_frame in bbox[1].keys():
                for i, bb in enumerate(bbox[1][n_frame]):
                    cv2.rectangle(frame2, pt1=(int(bb[0]), int(bb[1])),
                                  pt2=(int(bb[0]) + 50,
                                       int(bb[1]) + 50),
                                  color=(225, 225, 225), thickness=10)
                    # frame2 = put_mask(frame2, bb)

            frame = np.vstack((frame1, frame2))

            cv2.imshow(title, frame)
            frame = cv2.resize(frame, (frame_width, 2 * frame_height))
            if IS_SAVE: writer.write(frame)

            # cv2.waitKey(0)

            key = cv2.waitKey(100) & 0xFF

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
    for i in ['05', '06', '07', '08']:
        main(i)
