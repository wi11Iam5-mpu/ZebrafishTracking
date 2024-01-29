import datetime
import re
import subprocess
from pathlib import Path

import pandas as pd

result_path = Path(r'D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\outputs')
eval_path = Path(fr'D:\Projects\FishTracking\for_release\ZebrafishTracking\modules\utils\eval\mot_evaluation.py')


def run_terminal(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    curline = p.stdout.readline()
    while curline != b'':
        log_b = curline.replace(b"\r\n", b"")
        log_tmp = log_b.decode('utf-8', 'replace')
        print(log_tmp)
        if log_tmp.find('Error') != -1:
            print(log_tmp)
            break
        curline = p.stdout.readline()
    # p.wait()
    p.communicate()
    # print(p.returncode)


def get_metrics(SEQUENCES=None):
    if SEQUENCES is None:
        SEQUENCES = ["ZebraFish-01", "ZebraFish-02", "ZebraFish-03", "ZebraFish-04"]
    print(datetime.datetime.now())
    print("evaluation start ... ...")
    ####
    run_terminal(
        "cmd.exe /c" +
        # fr"conda activate motmetrics-env && "
        fr"python {str(eval_path)} --sequences {' '.join(SEQUENCES)}")
    print("evaluation done.")
    ####
    for data_index in SEQUENCES:
        metrics_file_path = result_path / fr"{''.join(re.findall(r'[0-9]', data_index))}"
        print(datetime.datetime.now())
        indexes = ["MOT_Metrics"]  # "MOT_Metrics", "MOT_Metrics_2d_cam1", "MOT_Metrics_2d_cam2"
        titles = ["3D"]  # "3D" "CAM1", "CAM2"
        for i, index in enumerate(indexes):
            df = pd.read_csv(Path(metrics_file_path) / 'metrics' / f'{index}.csv', sep=';')
            mota = (df['MOTA'].str.strip("%").astype(float)).values[0]
            idf1 = (df['ID F1-score'].str.strip("%").astype(float)).values[0]
            precision = (df['Precision'].str.strip("%").astype(float)).values[0]
            recall = (df['Recall'].str.strip("%").astype(float)).values[0]
            fp = (df['False Positives']).values[0]
            fn = (df['False Negatives']).values[0]
            ids = (df['Identity Swaps']).values[0]

            print(f"{data_index} {titles[i]} MOTA:[{mota: 0.2f}% ] "
                  f"IDF1:[{idf1: 0.2f}% ] "
                  f"precision:[{precision: 0.2f}% ] "
                  f"recall:[{recall: 0.2f}% ] "
                  f"fp:[{fp}] "
                  f"fn:[{fn}] "
                  f"ids:[{ids}]")


if __name__ == '__main__':
    get_metrics()
