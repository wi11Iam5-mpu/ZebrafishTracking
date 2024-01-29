import datetime
import subprocess
from pathlib import Path

import pandas as pd
# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval
# define a search space
from hyperopt import hp
from configs.config import configs, SEQUENCES

space = {
    'max_age': hp.quniform("p1", 5, 80, 5),
    # 'min_hits': hp.quniform("p2", 0, 1, 1),
    'cost_threshold': hp.quniform("p3", 0.1, 2, 0.1),
    'app_ratio_3d': hp.quniform("p7", 0.1, 0.5, 0.1),

    'min_err_thresh': hp.quniform("p4", 5, 80, 5),
    # 'max_err_thresh': hp.quniform("p5", 5, 80, 5),
    'max_app_thresh': hp.quniform("p6", 0.1, 0.9, 0.1),
}


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
    p.kill()
    p.wait()


# define an objective function
def objective(args):
    # 运行关联算法 & 评估结果
    print(
        f"args ==> "
        f"max_age {int(args['max_age'])} "
        # f"min_hits {int(args['min_hits'])} "
        f"cost_threshold {args['cost_threshold']} "
        f"app_ratio_3d {round(args['app_ratio_3d'], 2)} "
        f"min_err_thresh {int(args['min_err_thresh'])} "
        # f"max_err_thresh {int(args['max_err_thresh'])} "
        f"max_app_thresh {round(args['max_app_thresh'])} "
    )
    run_terminal(
        "cmd.exe /c" +
        fr"python D:\Projects\FishTracking\for_release\ZebrafishTracking\experiments\exp2-track.py "
        fr"--max_age {int(args['max_age'])} "
        # fr"--min_hits {int(args['min_hits'])} "
        fr"--cost_threshold {args['cost_threshold']} "
        fr"--app_ratio_3d {args['app_ratio_3d']} "
        fr"--min_err_thresh {args['min_err_thresh']} "
        # fr"--max_err_thresh {args['max_err_thresh']} "
        fr"--max_app_thresh {args['max_app_thresh']} "
        fr"--is_use_hyperopt"
    )

    metrics_mota = 0
    metrics_idf1 = 0

    with open(f'hyperopt_eval_record.txt', 'a+') as f:
        print(datetime.datetime.now(), file=f)
        print(args, file=f)
        seqs = [seq['index'] for seq in SEQUENCES]
        for data_index in seqs:
            metrics_file_path = fr"D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\outputs\{data_index}"
            csv_file = Path(metrics_file_path) / 'metrics' / 'MOT_Metrics.csv'
            df = pd.read_csv(csv_file, sep=';')
            mota = (df['MOTA'].str.strip("%").astype(float) / 100).values[0]
            idf1 = (df['ID F1-score'].str.strip("%").astype(float) / 100).values[0]
            precision = (df['Precision'].str.strip("%").astype(float) / 100).values[0]
            recall = (df['Recall'].str.strip("%").astype(float) / 100).values[0]
            fp = (df['False Positives']).values[0]
            fn = (df['False Negatives']).values[0]
            ids = (df['Identity Swaps']).values[0]

            print(f"{data_index} MOTA:[{mota * 100 : 0.2f}% ] "
                  f"IDF1:[{idf1 * 100 : 0.2f}% ] "
                  f"precision:[{precision * 100 : 0.2f}% ] "
                  f"recall:[{recall * 100 : 0.2f}% ] "
                  f"fp:[{fp}] "
                  f"fn:[{fn}] "
                  f"ids:[{ids}]", file=f)

            metrics_mota += mota
            metrics_idf1 += idf1

        metrics = metrics_mota + metrics_idf1
        print({'metrics': metrics, 'metrics_mota': metrics_mota, 'metrics_idf1': metrics_idf1}, file=f)
        print(f"metrics ==> metrics_mota {round(metrics_mota, 3)} "
              f"metrics_idf1 {round(metrics_idf1, 3)} "
              f"metrics {round(metrics, 3)}")

    return -metrics


def saft_tpe(seq=None):
    if seq is None:
        seq = ['01', '02', '03', '04']
    print(datetime.datetime.now())
    # 推荐：使用终端分窗口分别观察几个序列的评测结果
    print(f"======================{seq}========================")
    best = fmin(objective, space, algo=tpe.suggest, max_evals=1000)
    print("=======================================================")
    with open('hyperopt.txt', 'a+') as f:
        print(datetime.datetime.now(), file=f)
        print(best)
        print(f"{seq}, {space_eval(space, best)}", file=f)


if __name__ == '__main__':
    saft_tpe([seq['index'] for seq in SEQUENCES])
