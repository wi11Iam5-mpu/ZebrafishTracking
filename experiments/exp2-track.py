import argparse
import time

from configs.config import configs, SEQUENCES
from modules.data import ZebrafishSequence, ZebrafishDetSimple
from modules.trackers.Tracker import MHPTracker
from modules.utils.eval.evaluation import get_metrics
from modules.utils.misc import time_count


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=50)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=0)
    parser.add_argument("--cost_threshold", type=float, default=0.5)
    parser.add_argument("--app_ratio_3d", type=float, default=0.1)

    parser.add_argument("--min_err_thresh", type=float, default=15)
    parser.add_argument("--max_err_thresh", type=float, default=80)
    parser.add_argument("--max_app_thresh", type=float, default=0.7)

    parser.add_argument("--is_use_hyperopt", action="store_true", default=False)
    args = parser.parse_args()
    return args


@time_count(is_print=True)
def main(configures, args) -> None:
    start = time.time()
    seqs = []
    for cfg in configures:
        seqs.append(ZebrafishSequence(detector='pre_det',  # pre_det | yolo | gt
                                      det_path=cfg['detection_path'],
                                      img_path=cfg['image_path'],
                                      model_path=cfg['embedding_model_path']))

    print(f'Loading time: {time.time() - start:.1f} s')

    for index, cfg in enumerate(configures):
        if args.is_use_hyperopt:
            params = cfg["3d_tracker"]['params']
            params[0] = args.max_age
            params[1] = args.min_hits
            params[2] = args.cost_threshold
            params[3] = args.app_ratio_3d
            params = cfg["magic_numbers"]
            params['min_err_thresh'] = args.min_err_thresh
            params['max_err_thresh'] = args.max_err_thresh
            params['max_app_thresh'] = args.max_app_thresh

        eg = MHPTracker(**cfg)
        # eg.dotracking(seqs[index], tracker='mht')
        eg.dotracking(seqs[index], tracker='')

        del eg

    get_metrics(['ZebraFish-' + seq['index'] for seq in SEQUENCES])


if __name__ == '__main__':
    main(configs, parse_args())
