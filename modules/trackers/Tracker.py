import itertools
import time
from io import StringIO

import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
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


class MHPTracker(ConstructFirstMethod):

    def __init__(self, **cfg):
        super().__init__()
        self.calibration_path = cfg['calibration_path']
        self.output_path = cfg['output_path']
        # ...
        self.tracker_3d_max_age = cfg['3d_tracker']['params'][0]
        self.tracker_3d_min_hints = cfg['3d_tracker']['params'][1]
        self.tracker_3d_cost_threshold = cfg['3d_tracker']['params'][2]
        self.app_ratio_3d = cfg['3d_tracker']['params'][3]
        # ...
        self.min_err_thresh = cfg['magic_numbers']['min_err_thresh']
        self.max_err_thresh = cfg['magic_numbers']['max_err_thresh']
        self.max_app_thresh = cfg['magic_numbers']['max_app_thresh']
        # ...
        self.cam1_obj, self.cam2_obj = None, None
        self.fish_number = cfg['fish_number']
        self.load_calibrations()

    def load_calibrations(self):
        # Reading calibration weight
        self.cam1_obj = self.calibration_path / 'cam1.pkl'
        self.cam2_obj = self.calibration_path / 'cam2.pkl'

        self.cam1_obj = joblib.load(self.cam1_obj)
        self.cam1_obj.calcExtrinsicFromJson(self.calibration_path / 'cam1_references.json')

        self.cam2_obj = joblib.load(self.cam2_obj)
        self.cam2_obj.calcExtrinsicFromJson(self.calibration_path / 'cam2_references.json')

    @staticmethod
    def normalization(data):
        _range = np.max(data) - np.min(data)
        if _range != 0:
            return (data - np.min(data)) / _range
        else:
            return (data - np.min(data)) * _range

    @staticmethod
    def calc_embedding_distance(e1, e2, size=128):
        e1, e2 = torch.from_numpy(e1), torch.from_numpy(e2)
        e1, e2 = e1.view(-1, size), e2.view(-1, size)
        # return F.pairwise_distance(e1, e2, 2).numpy()[0]
        return 1 - (F.cosine_similarity(e1, e2).numpy() + 1) / 2

    @staticmethod
    def calc_embedding_sim(e1, e2, size=128):
        e1, e2 = torch.from_numpy(e1), torch.from_numpy(e2)
        e1, e2 = e1.view(-1, size), e2.view(-1, size)
        a = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        b = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
        return float(np.dot(a, b.T))

    def compute_similarity(self, detections, tracks):
        sim = {}
        _min, _max = np.inf, -np.inf
        app_th = 0.7  # dist = 1 - sim
        for f1, f2 in itertools.product(detections, tracks):
            embed1, embed2 = detections[f1]['app'], tracks[f2]['app']
            s1 = self.calc_embedding_distance(embed1[0], embed2[0])
            s2 = self.calc_embedding_distance(embed1[1], embed2[1])
            if np.any([s1 > app_th, s2 > app_th]):
                s1 = s2 = 1
            score = s1 + s2
            # _min = score if score < _min else _min
            # _max = score if score > _max else _max
            sim[(f1, f2)] = score / 2
        # _range = _max - _min
        # _range = 1 / _range if _range != 0 else _range
        for k in sim:
            sim[k] = 1 - sim[k]
            # sim[k] = 1 - (sim[k] - _min) * _range
            # sim[k] = sim[k] + 0.0001 if sim[k] == 0 else sim[k]
            # sim[k] = 1 - sim[k]
        return sim

    @time_count(is_print=False)
    # def dotracking(self, seq, outfile="tracks_3d_interpolated.csv", tracker='mht'):
    def dotracking(self, seq, outfile="tracks_3d.csv", tracker='mht'):

        def coordinate_interp(p1, p2, n: int):
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            if z2 < z1:
                return coordinate_interp(p2, p1, n)

            z = np.linspace(z1, z2, int(n))
            x = np.interp(z, (z1, z2), (x1, x2))
            y = np.interp(z, (z1, z2), (y1, y2))

            return zip(x, y, z)

        print(f"top / front detections: {len(seq.top_det)} / {len(seq.front_det)}")
        inters_frames = np.intersect1d(np.array(seq.top_frame), np.array(seq.front_frame))
        print(f"top / front frames: {len(seq.top_frame), len(seq.front_frame)}")
        print(f"top / front intersect frames: {len(inters_frames)}")
        err_profile = []
        app_profile = []

        if tracker != 'mht':
            # create instance of the 3D_SORT tracker
            mot_tracker = Sort3DWeighted(max_age=self.tracker_3d_max_age,
                                         min_hits=self.tracker_3d_min_hints,
                                         cost_threshold=self.tracker_3d_cost_threshold,
                                         app_item_weight=self.app_ratio_3d)
        else:
            # create instance of the 3D_MHT tracker
            params = {'K': 1, 'init_score': 1.2,
                      'kalman_constant_noise': True,
                      'kalman_Q_pos': 0.25,
                      'kalman_Q_vel': 1,
                      'kalman_R': 0.01,
                      'P_D': 0.9, 'P_FA': 0.00000048225,  # 0.00000048225,
                      'kin_null': 1,
                      'distance_threshold': 500,  # d^2 gating 1.0e-03
                      'canonical_kin_prob': True,
                      'max_scale_change': 500,  # x,y,z delta gating 0.5
                      'appearance_weight': 0.9, 'app_null': 0.3,
                      'max_missing': 10, 'min_track_quality': 0.1,
                      'min_track_length': 50, 'max_num_leaves': 5,
                      'use_gurobi': True, 'min_det_conf': -1.0}
            mht = MHTTracker(params)

        all_dets_string = [f"frame,id,3d_x,3d_y,3d_z,t_x,t_y,t_w,t_h,t_c,f_x,f_y,f_w,f_h,f_c\n"]
        record_string = [f"frame,id,3d_x,3d_y,3d_z\n"]
        det3d_string = [f"frame,id,3d_x,3d_y,3d_z\n"]
        start = time.time()
        frames = len(inters_frames)
        for frame in inters_frames:
            # print(f"\n===========Frame: {frame}===============")

            top_candidates = seq.top_dict[frame]
            front_candidates = seq.front_dict[frame]

            embedding_pairs = []
            integrated_points = []
            epipolar_err_matrix = []
            embedding_dist_matrix = []

            for c in itertools.product(top_candidates, front_candidates):
                top_det = c[0]
                front_det = c[1]

                top_det_point = (float(top_det.c_x), float(top_det.c_y))
                front_det_point = (float(front_det.c_x), float(front_det.c_y))

                top_det_bbox = (top_det.tl_x, top_det.tl_y, top_det.w, top_det.h)
                front_det_bbox = (front_det.tl_x, front_det.tl_y, front_det.w, front_det.h)

                top_det_embedding = c[0].embedding
                front_det_embedding = c[1].embedding

                embedding_distance = self.calc_embedding_distance(top_det_embedding,
                                                                  front_det_embedding, size=c[0].embedding.shape[0])

                _err, p = self.point3d_construct(top_det_point, front_det_point)

                integrated_points.append((p, (top_det_bbox, front_det_bbox)))
                epipolar_err_matrix.append(_err)
                embedding_dist_matrix.append(embedding_distance)
                embedding_pairs.append((top_det_embedding, front_det_embedding))

            err_profile += epipolar_err_matrix
            app_profile += embedding_dist_matrix  # embedding_dist_matrix is List

            # Mini-cost Hungary Matching
            err_cost_matrix = np.array(epipolar_err_matrix).reshape((len(top_candidates), len(front_candidates)))
            app_similar_matrix = np.array(embedding_dist_matrix).reshape(
                (len(top_candidates), len(front_candidates)))

            cost_matrix_err = np.copy(err_cost_matrix)
            cost_matrix_app = np.copy(app_similar_matrix)
            # cost_matrix_app = self.normalization(cost_matrix_app)
            # print(app_similar_matrix)

            # The points with small polarity error are used as the main criterion for appearance similarity.
            mask_o = np.zeros(np.array(cost_matrix_err).shape)
            mask_i = np.ones(np.array(cost_matrix_err).shape)
            indics = np.where(cost_matrix_err < self.min_err_thresh)
            mask_o[indics] = 1
            mask_i -= mask_o
            if indics[0].shape[0] > self.fish_number:
                cost_matrix = mask_o * cost_matrix_app * self.min_err_thresh + mask_i * cost_matrix_err
            else:
                cost_matrix = cost_matrix_err

            # match_indices = self.linear_assignment(cost_matrix_err)
            # match_indices = self.linear_assignment(cost_matrix_app)
            match_indices = self.linear_assignment(cost_matrix)

            # draw detection results
            detections = {}
            counter = 1
            dets = []
            for (row, col) in match_indices:
                index = row * match_indices.shape[0] + col
                p, bbox_pair = integrated_points[index]
                embeddings = embedding_pairs[index]

                err = err_cost_matrix.reshape(1, -1).tolist()[0][index]
                app = app_similar_matrix.reshape(1, -1).tolist()[0][index]
                # if self.is_pt_in_tank(*p) and float(err) * float(
                #         app) < self.max_err_thresh * self.max_app_thresh:
                if self.is_pt_in_tank(*p) and float(err) < self.max_err_thresh:
                    dets.append(Detection3D(coordinates=np.round(np.array(p), 2),
                                            boxes_pair=bbox_pair,
                                            thetas=(None, None),
                                            embeddings=embeddings,
                                            frame=frame))
                    det3d_string.append(
                        f"{int(frame)},-1,{float(np.round(p[0], 2))},{np.round(p[1], 2)},{np.round(p[2], 2)}\n")

                    #          x      y     z    c
                    # 'det': [p[0], p[1], p[2], 0.0, frame, counter, dummy]
                    #          0      1     2    3    4       5        6
                    detections[counter] = {'det': [p[0], p[1], p[2], 0.5] + [frame, counter, 0], 'app': embeddings}
                    counter += 1

            if len(dets) != 0 or counter > 1:
                if tracker != 'mht':
                    trackers = mot_tracker.update(dets)
                    for d in trackers:
                        res = f"{int(frame)},{int(d[3])},{float(d[0])},{float(d[1])},{float(d[2])}\n"
                        record_string.append(res)
                        all_dets_string.append(
                            f"{int(frame)},{int(d[3])},{float(d[0])},{float(d[1])},{float(d[2])},"
                            f"{float(d[4])},{float(d[5])},{float(d[6])-float(d[4])},{float(d[7])-float(d[5])},-1,"
                            f"{float(d[8])},{float(d[9])},{float(d[10])-float(d[8])},{float(d[11])-float(d[9])},-1\n"
                        )
                else:
                    features, _ = mht.getTrackPatches()
                    sim = self.compute_similarity(detections, features)
                    # print("dets: ", [np.round(detections[k]['det'][0:3], 3) for k in detections])
                    tracks = mht.doTracking(frame, detections, sim)
                    # print(mht.hypothesis_set.keys())

        if tracker == 'mht':
            mht.concludeTracks()
            _id = 1
            for t in mht.confirmed_tracks:
                # track:[frame, x, y, z, _, dummy, 0] t:id
                track_list = mht.confirmed_tracks[t]
                for track in track_list:
                    record_string.append('%d,%d,%.2f,%.2f,%.2f\n' % (track[0], _id, track[1], track[2], track[3]))
                _id = _id + 1

        end = time.time() - start

        with open(self.output_path / outfile, "w") as f:
            f.write(''.join(record_string))
        with open(self.output_path / "dets_3d.csv", "w") as f:
            f.write(''.join(det3d_string))
        with open(self.output_path / "all_dets.csv", "w") as f:
            f.write(''.join(all_dets_string))

        print(f"Total Time:{end:.3f} s, FPS: {frames / end :.1f}")

        if True:
            # result_3d = pd.read_csv(StringIO(''.join(record_string)), sep=',')
            result_3d = pd.read_csv(self.output_path / "tracks_3d.csv", sep=',')
            result_3d.columns = ['frame', 'id', '3d_x', '3d_y', '3d_z']
            result_3d['frame'] = result_3d['frame'].astype(int)
            result_3d = result_3d.sort_values(by=['id', 'frame'])
            ids = result_3d['id'].unique()
            df_interp = defaultdict(list)
            for index in ids:
                df_id = result_3d[result_3d.id == index]
                frames = df_id['frame'].unique()
                all_frames = np.linspace(np.min(frames), np.max(frames),
                                         np.max(frames) - np.min(frames) + 1)

                miss_frames = sorted(set(all_frames) - set(frames))
                tmp = defaultdict(list)

                miss_index = 0
                for fr in miss_frames:
                    tmp[miss_index].append(fr)
                    if fr + 1 not in miss_frames:
                        miss_index += 1

                for k, v in tmp.items():
                    frame_start, frame_end = v[0], v[-1]
                    point1 = df_id[df_id.frame == frame_start - 1][['3d_x', '3d_y', '3d_z']].values
                    point2 = df_id[df_id.frame == frame_end + 1][['3d_x', '3d_y', '3d_z']].values
                    points = coordinate_interp(*point1, *point2, frame_end - frame_start + 1)
                    for i, p in enumerate(points):
                        df_interp['frame'].append(frame_start + i)
                        df_interp['id'].append(index)
                        df_interp['3d_x'].append(p[0])
                        df_interp['3d_y'].append(p[1])
                        df_interp['3d_z'].append(p[2])

            miss_interp_df = pd.DataFrame(data=df_interp)
            result_3d = result_3d[['frame', 'id', '3d_x', '3d_y', '3d_z']]
            final_df = pd.concat([result_3d, miss_interp_df], axis=0, ignore_index=True)
            final_df = final_df.sort_values(by=['frame', 'id'])
            final_df.to_csv(self.output_path / r"tracks_3d_interpolated.csv", sep=',', index=False)

    def point3d_construct(self, point1, point2):
        # 1) Triangulate 3D point
        tr = Triangulate()
        p, d = tr.triangulatePoint(point1,
                                   point2,
                                   self.cam1_obj,
                                   self.cam2_obj,
                                   correctRefraction=True)
        p1 = self.cam1_obj.forwardprojectPoint(*p)
        p2 = self.cam2_obj.forwardprojectPoint(*p)
        # 2) Calc re-projection errors
        pos1 = np.array(point1)
        err1 = np.linalg.norm(pos1 - p1)
        pos2 = np.array(point2)
        err2 = np.linalg.norm(pos2 - p2)
        err = err1 + err2
        return err, p

    @staticmethod
    def is_pt_in_tank(x, y, z):
        if x < 0 or x > 29: return False
        if y < 0 or y > 29: return False
        if z < 0 or z > 29: return False
        return True

    @staticmethod
    def linear_assignment(cost_matrix):
        try:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))
        except ImportError:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i], i] for i in x if i >= 0])
