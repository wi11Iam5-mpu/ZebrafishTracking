import os
from collections import deque, defaultdict
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IS_SAVE = False

colors = [

    u'#AD2123',
    u'#7D5C5D',
    u'#DF41E0',
    u'#C3E472',
    u'#B3AF07',

    u'#5421AD',
    u'#685C7D',
    u'#415DE0',
    u'#E4BC72',
    u'#B36103',

    u'#216FAD',
    u'#5C6E7D',
    u'#41E0D3',
    u'#E48772',
    u'#B30F0A',

    u'#21AD63',
    u'#5C7D6C',
    u'#54E041',
    u'#DA72E4',
    u'#7400B8',

    u'#AD6F21',
    u'#7D6F5C',
    u'#E06B41',
    u'#72E4AD',
    u'#03B32B',

]


def Hex_to_RGB(hex):
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    return (r, g, b)

# show max_len=200 frame
index = '06'

colors = [Hex_to_RGB(c) for c in colors]


def main(seq_index, method):
    title = '3d'
    fps = 60
    code = 'MP4V'
    videos = ['cam1', 'cam2']
    input_path = Path(
        r"/sequences")

    if method == 'naive':
        show_trk = fr'D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\baseline\naive\ZebraFish-{seq_index}.txt'
    elif method == 'gt':
        show_trk = fr'D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\baseline\gt\ZebraFish-{seq_index}.txt'
    elif method == 'mvi':
        show_trk = fr'D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\baseline\mvi\ZebraFish-{seq_index}.txt'
    elif method == 'our':
        show_trk = fr'D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\baseline\our\ZebraFish-{seq_index}.txt'
    else:
        show_trk = input_path / 'outputs' / seq_index / 'all_dets.csv'

    #

    raw_video1 = input_path / 'videos' / seq_index / 'cam1.mp4'
    raw_video2 = input_path / 'videos' / seq_index / 'cam2.mp4'
    output_video_name = input_path / 'videos' / seq_index / 'tracks_3d.mp4'

    raw_video_path = [raw_video1, raw_video2]

    if 'baseline' in str(show_trk):
        data_pack = pd.read_csv(show_trk, header=None)
        data_pack.columns = ['frame', 'id', '3d_x', '3d_y', '3d_z', 't_xc', 't_yc', 't_x', 't_y', 't_w', 't_h', 't_c',
                             'f_xc', 'f_yc', 'f_x', 'f_y', 'f_w', 'f_h',
                             'f_c']
        selected = [['frame', 't_xc', 't_yc', 't_w', 't_h', 'id'],
                    ['frame', 'f_xc', 'f_yc', 'f_w', 'f_h', 'id']]
    else:
        data_pack = pd.read_csv(show_trk)
        selected = [['frame', 't_x', 't_y', 't_w', 't_h', 'id'],
                    ['frame', 'f_x', 'f_y', 'f_w', 'f_h', 'id']]

    ids = list(set(data_pack['id'].values))

    new_ids = dict(zip(ids, list(range(len(ids)))))

    # set color
    # colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(len(ids))]

    data_pack_selected = [data_pack[labels].values for labels in selected]

    # clrs_n = len(ids)
    # colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #           for _ in range(clrs_n)]

    readers = [cv2.VideoCapture(str(p)) for p in raw_video_path]

    frame_count = int(readers[0].get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(readers[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(readers[0].get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*code)
    writer = cv2.VideoWriter(str(output_video_name), fourcc, fps, (frame_width, 2 * frame_height))

    s_factor = 3
    set_frame_width = int(frame_width // s_factor)
    set_frame_height = int(frame_height // s_factor)

    bbox = [defaultdict(list), defaultdict(list)]

    for i, data in enumerate(data_pack_selected):
        for d in data:
            if 'baseline' in str(show_trk):
                bbox[i][int(d[0])].append(d[1:])
            else:
                bbox[i][int(d[0])].append([d[1] + d[3] / 2, d[2] + d[4] / 2, d[3], d[4], d[5]])

    cv2.namedWindow(title, cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(title, set_frame_width // 1, 2 * set_frame_height // 1)

    while readers[0].isOpened() and readers[1].isOpened():
        max_len = 200
        q_f = deque(maxlen=max_len)
        q_t = deque(maxlen=max_len)
        for n_frame in range(1, frame_count + 1):
            ret, frame1 = readers[0].read()
            ret, frame2 = readers[1].read()

            cv2.putText(frame1, f'frame: {n_frame}', (100, 150), cv2.FONT_HERSHEY_PLAIN, 5, color=(0, 0, 255),
                        thickness=3)

            if n_frame in bbox[0].keys():
                for i, bb in enumerate(bbox[0][n_frame]):

                    cv2.putText(frame1, str(new_ids[int(bb[4])]), (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_PLAIN, 3,
                                color=colors[new_ids[int(bb[4])]],
                                thickness=3)

                    # plot the 2d trajectory
                    cv2.circle(frame1, (int(bb[0]), int(bb[1])), 5,
                               color=colors[new_ids[int(bb[4])]],
                               thickness=-1)
                    if len(q_t) > 0:
                        for i in range(1, len(q_t)):
                            if q_t[i - 1] is None or q_t[i] is None:
                                continue
                            thickness = int(np.power(1 / float(i / 3 + 1), 0.3) * 20)
                            cv2.circle(frame1, (q_t[-i][0], q_t[-i][1]), thickness, color=q_t[-i][2], thickness=-1)

                    if len(q_t) < max_len:
                        q_t.append((int(bb[0]), int(bb[1]), colors[new_ids[int(bb[4])]]))
                    else:
                        q_t.popleft()
                        q_t.append((int(bb[0]), int(bb[1]), colors[new_ids[int(bb[4])]]))
            else:
                if len(q_t) > 0:
                    for i in range(1, len(q_t)):
                        if q_t[i - 1] is None or q_t[i] is None:
                            continue
                        thickness = int(np.power(1 / float(i / 3 + 1), 0.3) * 20)
                        cv2.circle(frame1, (q_t[-i][0], q_t[-i][1]), thickness, color=q_t[-i][2], thickness=-1)

            if n_frame in bbox[1].keys():
                for i, bb in enumerate(bbox[1][n_frame]):

                    cv2.putText(frame2, str(new_ids[int(bb[4])]), (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_PLAIN, 5,
                                color=colors[new_ids[int(bb[4])]],
                                thickness=3)

                    cv2.circle(frame2, (int(bb[0]), int(bb[1])), 5,
                               color=colors[new_ids[int(bb[4])]],
                               thickness=-1)
                    if len(q_f) > 0:
                        for i in range(1, len(q_f)):
                            if q_f[i - 1] is None or q_f[i] is None:
                                continue
                            # thickness = int(np.power(32 / float(i / 3 + 1), 0.3) * 2.5)
                            thickness = int(np.power(1 / float(i / 3 + 1), 0.3) * 20)
                            cv2.circle(frame2, (q_f[-i][0], q_f[-i][1]), thickness, color=q_f[-i][2], thickness=-1)
                    if len(q_f) < max_len:
                        q_f.append((int(bb[0]), int(bb[1]), colors[new_ids[int(bb[4])]]))

                    else:
                        q_f.popleft()
                        q_f.append((int(bb[0]), int(bb[1]), colors[new_ids[int(bb[4])]]))
            else:
                if len(q_f) > 0:
                    for i in range(1, len(q_f)):
                        if q_f[i - 1] is None or q_f[i] is None:
                            continue
                        # thickness = int(np.power(32 / float(i / 3 + 1), 0.3) * 2.5)
                        thickness = int(np.power(1 / float(i / 3 + 1), 0.3) * 20)
                        cv2.circle(frame2, (q_f[-i][0], q_f[-i][1]), thickness, color=q_f[-i][2], thickness=-1)
            frame = np.vstack((frame1, frame2))

            if 'baseline' in str(show_trk):
                path_fig = fr'D:\tmp\{method}\{seq_index}\{n_frame}.png'
            else:
                path_fig = fr'D:\tmp\our\{seq_index}\{n_frame}.png'

            cv2.imwrite(path_fig, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

            cv2.imshow(title, frame)
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
    # for m in ['naive', 'our']:
    for m in ['our']:
        # for idx in ['01', '02', '03', '04']:
        for idx in ['02', '04']:
        # for idx in ['01', '02', '03', '04', '05', '06', '07', '08']:
            main(idx, m)
