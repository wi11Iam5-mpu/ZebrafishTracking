from collections import deque, defaultdict
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

random.seed(20210727)


def main(seq_index='01'):
    videos = ['cam1', 'cam2']
    input_path = Path(
        r"D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences")

    raw_video1 = input_path / 'videos' / seq_index / 'cam1.mp4'
    raw_video2 = input_path / 'videos' / seq_index / 'cam2.mp4'

    # show_trk = input_path / 'outputs' / seq_index / 'tracks_3d.csv'
    show_trk = input_path / 'outputs' / seq_index / 'tracks_3d_interpolated.csv'

    IS_SAVE = True
    title = '3d'
    fps = 60
    code = 'MP4V'
    output_video_name = f'./res3d_{seq_index}.mp4'
    raw_video_path = [raw_video1, raw_video2]
    data_pack = pd.read_csv(show_trk)
    ids = list(set(data_pack['id'].values))

    selected = [['frame', 'id', 'top_x', 'top_y'],
                ['frame', 'id', 'front_x', 'front_y']]

    data_pack_selected = [data_pack[labels].values for labels in selected]

    clrs_n = len(ids) + 1000
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for _ in range(clrs_n)]

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

            cv2.putText(frame1, f'frame: {n_frame}', (100, 150), cv2.FONT_HERSHEY_PLAIN, 3, color=(0, 0, 255),
                        thickness=3)

            related = defaultdict(list)
            if n_frame in bbox[0].keys():
                for i, bb in enumerate(bbox[0][n_frame]):
                    # cv2.rectangle(frame1, pt1=(int(bb[0]), int(bb[1])),
                    #               pt2=(int(bb[0]) + int(bb[2]),
                    #                    int(bb[1]) + int(bb[3])),
                    #               color=colors[int(bb[4])], thickness=10)
                    cv2.circle(frame1, (int(bb[1]), int(bb[2])), 10, color=colors[int(bb[0])], thickness=-1)
                    cv2.putText(frame1, str(int(bb[0])), (int(bb[1]), int(bb[2])), cv2.FONT_HERSHEY_PLAIN, 3,
                                color=colors[int(bb[0])],
                                thickness=3)
                    # cv2.circle(frame1, (int(bb[0]), int(bb[1])), 5, color=colors[int(bb[4])], thickness=-1)
                    # if len(q) > 0:
                    #     for i in range(1, len(q)):
                    #         if q[i - 1] is None or q[i] is None:
                    #             continue
                    #         thickness = int(np.power(32 / float(i / 3 + 1), 0.3) * 2.5)
                    #         cv2.circle(frame1, q[-i], thickness, color=colors[int(bb[4])], thickness=-1)
                    # if len(q) < max_len:
                    #     q.append((int(bb[0]), int(bb[1])))
                    # else:
                    #     q.popleft()
                    #     q.append((int(bb[0]), int(bb[1])))

                    # related[i].append(
                    #     (
                    #         (int(bb[0]),
                    #          int(bb[1]) + int(bb[3])),
                    #         colors[int(bb[4])]
                    #     )
                    # )

            if n_frame in bbox[1].keys():
                for i, bb in enumerate(bbox[1][n_frame]):
                    # cv2.rectangle(frame2, pt1=(int(bb[0]), int(bb[1])),
                    #               pt2=(int(bb[0]) + int(bb[2]),
                    #                    int(bb[1]) + int(bb[3])),
                    #               color=colors[int(bb[4])], thickness=10)
                    cv2.circle(frame2, (int(bb[1]), int(bb[2])), 10, color=colors[int(bb[0])], thickness=-1)
                    cv2.putText(frame2, str(int(bb[0])), (int(bb[1]), int(bb[2])), cv2.FONT_HERSHEY_PLAIN, 3,
                                color=colors[int(bb[0])],
                                thickness=3)
                    # cv2.circle(frame2, (int(bb[0]), int(bb[1])), 5, color=colors[int(bb[4])], thickness=-1)
                    # if len(q) > 0:
                    #     for i in range(1, len(q)):
                    #         if q[i - 1] is None or q[i] is None:
                    #             continue
                    #         thickness = int(np.power(32 / float(i / 3 + 1), 0.3) * 2.5)
                    #         cv2.circle(frame2, q[-i], thickness, color=colors[int(bb[4])], thickness=-1)
                    # if len(q) < max_len:
                    #     q.append((int(bb[0]), int(bb[1])))
                    # else:
                    #     q.popleft()
                    #     q.append((int(bb[0]), int(bb[1])))

                    # related[i].append(
                    #     (
                    #         (int(bb[0]), int(bb[1])),
                    #         colors[int(bb[4])]
                    #     )
                    # )

            frame = np.vstack((frame1, frame2))

            # for k, pair in related.items():
            #     if len(pair) == 2:
            #         ptStart = (pair[0][0][0], pair[0][0][1])
            #         ptEnd = (pair[1][0][0], pair[1][0][1] + frame.shape[0] // 2)
            #         point_color = pair[0][1]  # BGR
            #         thickness = 2
            #         lineType = 4
            #         cv2.line(frame, ptStart, ptEnd, point_color, thickness, lineType)
            # show_frame = cv2.resize(frame, set_frame_width, set_frame_height)
            cv2.imshow(title, frame)
            frame = cv2.resize(frame, (frame_width, 2 * frame_height))
            if IS_SAVE: writer.write(frame)
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
    # for i in ['05', '06', '07', '08']:
    for i in ['03']:
        main(i)
