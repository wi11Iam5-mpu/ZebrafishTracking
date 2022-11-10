from pathlib import Path

import cv2
import os

DIR_BASE = Path(r"D:\Datasets\3DZeF20")

# file = 'ZebraFish-07'
# files = ['ZebraFish-05', 'ZebraFish-06', 'ZebraFish-07', 'ZebraFish-08']
files = ['ZebraFish-01', 'ZebraFish-02', 'ZebraFish-03', 'ZebraFish-04']
file = files[3]

# pfile = 'test'
pfile = 'train'

for cam_type in ['F', 'T']:
    out_folder = DIR_BASE / f'{pfile}/{file}'
    image_folder = DIR_BASE / f'{pfile}/{file}/img{cam_type}'

    if cam_type == 'T':
        video_name = f'{out_folder}/cam1.mp4'
    elif cam_type == 'F':
        video_name = f'{out_folder}/cam2.mp4'
    # video_name = f'{file}_img{cam_type}.mp4'

    fps = 60
    # code = 'XVID'  # .avi
    code = 'MP4V'  # .mp4  tag 0x34504d46/'FMP4' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
    # code = 'FMP4'  # .mp4  tag 0x34504d46/'FMP4' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
    # code = 'X264'  # .mp4 tag 0x34363258/'X264' is not supported with codec id 27 and format 'mp4 / MP4 (MPEG-4 Part 14)'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    print(f'Converting {file} {cam_type}')
    print(height, width, layers)

    fourcc = cv2.VideoWriter_fourcc(*code)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
