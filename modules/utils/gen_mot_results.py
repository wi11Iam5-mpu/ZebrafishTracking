src_files = [
    "tracks_3d_interpolated-01.csv"
    , "tracks_3d_interpolated-02.csv"
    , "tracks_3d_interpolated-03.csv"
    , "tracks_3d_interpolated-04.csv"
    , "tracks_3d_interpolated-05.csv"
    , "tracks_3d_interpolated-06.csv"
    , "tracks_3d_interpolated-07.csv"
    , "tracks_3d_interpolated-08.csv"]

import pandas as pd

for i, file in enumerate(src_files):
    df = pd.read_csv(file, sep=',')
    df['frame'] = df.frame.astype(int)
    df['3d_x'] = df['3d_x'].round(3)
    df['3d_y'] = df['3d_y'].round(3)
    df['3d_z'] = df['3d_z'].round(3)
    df.to_csv(fr"ZebraFish-0{i + 1}.txt", header=None, index=None)
