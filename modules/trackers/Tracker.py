import itertools
import time

import cv2
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from packaging import version

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


class FishTracker(ConstructFirstMethod):

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

        self.err_big = 1000

    def load_calibrations(self):
        # Reading calibration weight
        self.cam1_obj = self.calibration_path / 'cam1.pkl'
        self.cam2_obj = self.calibration_path / 'cam2.pkl'

        self.cam1_obj = joblib.load(self.cam1_obj)
        self.cam1_obj.calcExtrinsicFromJson(self.calibration_path / 'cam1_references.json')

        self.cam2_obj = joblib.load(self.cam2_obj)
        self.cam2_obj.calcExtrinsicFromJson(self.calibration_path / 'cam2_references.json')

    @staticmethod
    def calc_embedding_distance(e1, e2, size=128):
        e1, e2 = torch.from_numpy(e1), torch.from_numpy(e2)
        e1, e2 = e1.view(-1, size), e2.view(-1, size)
        return 1 - (F.cosine_similarity(e1, e2).numpy() + 1) / 2

    @time_count(is_print=False)
    def dotracking(self, seq, outfile="tracks_3d.csv", tracker=''):

        def coordinate_interp(p1, p2, n: int):
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            if z2 < z1:
                return coordinate_interp(p2, p1, n)

            z = np.linspace(z1, z2, int(n))
            x = np.interp(z, (z1, z2), (x1, x2))
            y = np.interp(z, (z1, z2), (y1, y2))

            return zip(x, y, z)

        print(f"detection path: {seq.det_path}")
        print(f"output path: {self.output_path}")
        print(f"top / front detections: {len(seq.top_det)} / {len(seq.front_det)}")
        inters_frames = np.intersect1d(np.array(seq.top_frame), np.array(seq.front_frame))
        print(f"top / front frames: {len(seq.top_frame), len(seq.front_frame)}")
        print(f"top / front intersect frames: {len(inters_frames)}")

        # create instance of the 3D_SORT tracker
        mot_tracker = Sort3DWeighted(max_age=self.tracker_3d_max_age,
                                     min_hits=self.tracker_3d_min_hints,
                                     cost_threshold=self.tracker_3d_cost_threshold,
                                     app_item_weight=self.app_ratio_3d)

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

            # here can be simply by point3d_construct_batch()
            # need to modify the structure of detection dict.
            ############################################################################################################
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
                                                                  front_det_embedding,
                                                                  size=c[0].embedding.shape[0])

                _err, p = self.point3d_construct(top_det_point, front_det_point)
                # only valid points
                if self.is_pt_in_tank(*p):
                    integrated_points.append((p, (top_det_bbox, front_det_bbox)))
                    epipolar_err_matrix.append(_err)
                    embedding_dist_matrix.append(embedding_distance)
                    embedding_pairs.append((top_det_embedding, front_det_embedding))
                else:
                    integrated_points.append((p, (top_det_bbox, front_det_bbox)))
                    epipolar_err_matrix.append(self.err_big)
                    embedding_dist_matrix.append(np.array([1]))
                    embedding_pairs.append((top_det_embedding, front_det_embedding))

            ############################################################################################################

            # Mini-cost Hungary Matching
            err_cost_matrix = np.array(epipolar_err_matrix).reshape((len(top_candidates),
                                                                     len(front_candidates)))
            app_similar_matrix = np.array(embedding_dist_matrix).reshape((len(top_candidates),
                                                                          len(front_candidates)))

            cost_matrix_err = np.copy(err_cost_matrix)
            cost_matrix_app = np.copy(app_similar_matrix)

            # The points with small polarity error are used as the main criterion for appearance similarity.
            mask_o = np.zeros(np.array(cost_matrix_err).shape)
            mask_i = np.ones(np.array(cost_matrix_err).shape)
            indics = np.where(cost_matrix_err < self.min_err_thresh)
            cost_matrix_err[np.where(cost_matrix_err > self.max_err_thresh)] = self.err_big
            mask_o[indics] = 1
            mask_i -= mask_o
            cost_matrix = mask_o * cost_matrix_app * self.min_err_thresh + mask_i * cost_matrix_err
            # if indics[0].shape[0] > self.fish_number:
            #     cost_matrix = mask_o * cost_matrix_app * self.min_err_thresh + mask_i * cost_matrix_err
            # else:
            #     cost_matrix = cost_matrix_err

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
                if self.is_pt_in_tank(*p) and \
                        float(err) < self.max_err_thresh and \
                        float(app) < self.max_app_thresh:
                    dets.append(Detection3D(coordinates=np.round(np.array(p), 2),
                                            boxes_pair=bbox_pair,
                                            thetas=(None, None),
                                            embeddings=embeddings,
                                            frame=frame))
                    det3d_string.append(
                        f"{int(frame)},-1,{float(np.round(p[0], 2))},{np.round(p[1], 2)},{np.round(p[2], 2)}\n")
                    counter += 1

            if len(dets) != 0 or counter > 1:
                trackers = mot_tracker.update(dets)
                for d in trackers:
                    res = f"{int(frame)},{int(d[3])},{float(d[0])},{float(d[1])},{float(d[2])}\n"
                    record_string.append(res)
                    all_dets_string.append(
                        f"{int(frame)},{int(d[3])},{float(d[0])},{float(d[1])},{float(d[2])},"
                        f"{float(d[4])},{float(d[5])},{float(d[6]) - float(d[4])},{float(d[7]) - float(d[5])},-1,"
                        f"{float(d[8])},{float(d[9])},{float(d[10]) - float(d[8])},{float(d[11]) - float(d[9])},-1\n"
                    )

        end = time.time() - start

        with open(self.output_path / outfile, "w") as f:
            f.write(''.join(record_string))
        with open(self.output_path / "dets_3d.csv", "w") as f:
            f.write(''.join(det3d_string))
        with open(self.output_path / "all_dets.csv", "w") as f:
            f.write(''.join(all_dets_string))

        print(f"Total Time:{end:.3f} s, FPS: {frames / end :.1f}")

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
        use_cRefraction = True
        p, d = tr.triangulatePoint(point1,
                                   point2,
                                   self.cam1_obj,
                                   self.cam2_obj,
                                   correctRefraction=use_cRefraction)
        p1 = self.cam1_obj.forwardprojectPoint(*p, correctRefraction=use_cRefraction)
        p2 = self.cam2_obj.forwardprojectPoint(*p, correctRefraction=use_cRefraction)
        # 2) Calc re-projection errors
        pos1 = np.array(point1)
        err1 = np.linalg.norm(pos1 - p1)
        pos2 = np.array(point2)
        err2 = np.linalg.norm(pos2 - p2)
        err = err1 + err2
        return err, p

    def point3d_construct_batch(self, point1, point2):
        """
        implement point3d_construct() with a batch way
        points shape (2,N,2)
        """
        # 1) Triangulate 3D point
        # 1) Backprojects points into 3D ray
        def backprojectPoint(points, K, dist, rot, pos):
            # Calculate R = K^-1 [x y 1]^T and account for distortion
            # points [[[]]] 1*N*2
            ray = cv2.undistortPoints(points, K, dist)
            ray = np.concatenate((ray[0], np.ones((points.shape[1], 1))), axis=1)

            # Calculate R^-1 R array([-0.20256649, -1.03068575, -0.01979361])
            ray = np.dot(np.linalg.inv(rot), ray.T).T
            ray /= np.linalg.norm(ray, axis=1).reshape(points.shape[1], -1)

            # Calculate camera center, i.e. -R^-1 t
            ray0 = pos
            return ray, ray0

        # ray1[0][0] array([-0.15791207,  0.27992322,  0.94694602])
        # ray2[0][0] array([-0.19281223, -0.98105478, -0.01884048])
        ray1 = backprojectPoint(point1, self.cam1_obj.K, self.cam1_obj.dist, self.cam1_obj.rot, self.cam1_obj.pos)
        ray2 = backprojectPoint(point2, self.cam2_obj.K, self.cam2_obj.dist, self.cam2_obj.rot, self.cam2_obj.pos)

        # 2) Find plane intersection
        def intersectionWithRay(r, r0, plane_normal, plane_points):
            n0 = plane_points[0]
            t = np.dot((n0 - r0), plane_normal)
            t = np.repeat(t, r.shape[0], axis=0)
            t /= np.dot(r, plane_normal)
            intersection = (t.reshape(r.shape[0], 1) * r) + r0
            return intersection

        p1Intersect = intersectionWithRay(*ray1, self.cam1_obj.plane.normal, self.cam1_obj.plane.points)
        p2Intersect = intersectionWithRay(*ray2, self.cam2_obj.plane.normal, self.cam2_obj.plane.points)

        # 3) Refract the backprojected rays
        def refractRay(rayDir, planeNormal, n1, n2):
            r = n1 / n2
            normPlane = planeNormal / np.linalg.norm(planeNormal)
            normDir = rayDir / np.linalg.norm(rayDir, axis=1).reshape(-1, 1)
            c1 = np.dot(-normPlane.reshape(1, 3), normDir.T).T
            c2 = np.sqrt(1.0 - np.power(r, 2) * (1.0 - np.power(c1, 2)))
            refracted = r * rayDir + (r * c1 - c2) * normPlane
            return refracted, c1, c2

        n1 = 1.0  # Refraction index for air
        n2 = 1.33  # Refraction index for water
        ref1, _, _ = refractRay(ray1[0], self.cam1_obj.plane.normal, n1, n2)
        ref2, _, _ = refractRay(ray2[0], self.cam2_obj.plane.normal, n1, n2)

        # 4) Triangulate points the refracted rays
        def rayIntersection(ray1Dir, ray1Point, ray2Dir, ray2Point):
            a = ray1Dir
            b = ray2Dir
            A = ray1Point
            B = ray2Point
            c = B - A

            ab = np.dot(a, b.T).diagonal()
            aa = np.dot(a, a.T).diagonal()
            bb = np.dot(b, b.T).diagonal()
            ac = np.dot(a, c.T).diagonal()
            bc = np.dot(b, c.T).diagonal()

            denom = aa * bb - ab * ab
            tD = (-ab * bc + ac * bb) / denom
            tE = (ab * ac - bc * aa) / denom

            D = A + a * tD.reshape(-1, 1)
            E = B + b * tE.reshape(-1, 1)
            point = (D + E) / 2
            dist = np.linalg.norm(D - E, axis=1)
            return point, dist

        rayIntersection = rayIntersection(ref1, p1Intersect, ref2, p2Intersect)
        p, d = rayIntersection[0], rayIntersection[1]

        def get_ref_point_batch(points, pos, plane_normal, plane_points):
            """ water2air forward project"""

            def inner_by_row(a, b):
                return np.array(list(map(lambda x: np.dot(x[0], x[1]), zip(a, b))))

            c1 = pos.squeeze()
            w = plane_normal

            # 1) Plane between points and c1, perpendicular to w
            n = np.cross((points - c1), w, axisa=1, axisb=0)

            # 2) Find plane origin and x/y directions
            #    i.e. project camera position onto refraction plane
            n0 = plane_points[0]
            t = np.dot((n0 - c1), w)
            t /= np.dot(-w, w)
            intersection = (t * -w) + c1

            p0 = intersection.reshape(1, -1)

            pX = c1 - p0
            pX = pX / np.linalg.norm(pX, axis=1, keepdims=True)
            pY = np.cross(n, pX)
            pY = pY / np.linalg.norm(pY, axis=1, keepdims=True)

            # 3) Project 3d positions and camera position onto 2D plane
            p1_proj = np.vstack([np.dot(pX, (points - p0).T),
                                 np.reshape(inner_by_row(pY, (points - p0)), (1, -1))]).T
            c1_proj = np.hstack([np.repeat(np.dot(pX, (c1 - p0).T), pY.shape[0], axis=0),
                                 np.dot(pY, (c1 - p0).T)])

            p1_proj = np.reshape(p1_proj, (-1, 2))

            # 4) Construct 4'th order polynomial
            sx = p1_proj[:, 0]
            sy = p1_proj[:, 1]
            e = c1_proj[:, 0]
            r = 1.33
            N = (1 / r ** 2) - 1

            coeffs = [
                [
                    N,  # y4
                    -2 * N * sy[i],  # y3
                    N * sy[i] ** 2 + (sx[i] ** 2 / r ** 2) - e[i] ** 2,  # y2
                    2 * e[i] ** 2 * sy[i],  # y1
                    -e[i] ** 2 * sy[i] ** 2  # y0
                ]
                for i in range(e.shape[0])
            ]
            res = np.vstack([np.roots(c) for c in coeffs])

            real = np.real(res)
            resRange = [(min(1e-6, sy[i]), max(1e-6, sy[i])) for i in range(sy.shape[0])]
            finalRes = []
            for i in range(real.shape[0]):
                valid = (real[i] > resRange[i][0]) & (real[i] < resRange[i][1])
                if np.any(valid):
                    validRes = real[i][valid]
                    finalRes.append(validRes[np.argmax(np.abs(validRes))])

            refPoints = np.array(finalRes)[:, None] * pY + p0
            return refPoints

        def project_points(points, extr_mat, intr_mat, dist):
            # Stack the points into a 2D tensor
            # points = torch.tensor(points, dtype=torch.float32)
            # convert point to homogeneous coordinates [x,y,z,1]
            ones = np.ones((points.shape[0], 1), dtype=np.float32)
            points = np.hstack((points, ones))

            # Project the points
            _p = np.matmul(extr_mat, points.T).T
            _pp = _p[:, :2] / _p[:, 2:]
            x, y = _pp[:, 0], _pp[:, 1]
            r2 = x ** 2 + y ** 2
            r4 = r2 ** 2
            r6 = r2 ** 3
            rdist = (1 + dist[0] * r2 + dist[1] * r4 + dist[4] * r6) / (1 + dist[5] * r2 + dist[6] * r4 + dist[7] * r6)
            x_dist = x * rdist
            y_dist = y * rdist
            tanx = 2 * dist[2] * x * y + dist[3] * (r2 + 2 * x ** 2)
            tany = dist[2] * (r2 + 2 * y ** 2) + 2 * dist[3] * x * y
            x_dist = x_dist + tanx
            y_dist = y_dist + tany

            # Back to absolute coordinates
            x_dist = intr_mat[0][0] * x_dist + intr_mat[0][2]
            y_dist = intr_mat[1][1] * y_dist + intr_mat[1][2]
            points_2d = np.vstack((x_dist, y_dist)).T

            return points_2d.astype(np.int32)

        # forward projection ...
        refPoints_top = get_ref_point_batch(p, self.cam1_obj.pos, self.cam1_obj.plane.normal,
                                            self.cam1_obj.plane.points)
        refPoints_side = get_ref_point_batch(p, self.cam2_obj.pos, self.cam2_obj.plane.normal,
                                             self.cam2_obj.plane.points)

        p1 = project_points(refPoints_top, self.cam1_obj.getExtrinsicMat(), self.cam1_obj.K,
                            self.cam1_obj.dist)
        p2 = project_points(refPoints_side, self.cam2_obj.getExtrinsicMat(), self.cam2_obj.K,
                            self.cam2_obj.dist)

        # 2) Calc re-projection errors
        pos1 = np.array(point1)
        err1 = np.linalg.norm(pos1 - p1, axis=2)
        pos2 = np.array(point2)
        err2 = np.linalg.norm(pos2 - p2, axis=2)
        err = err1 + err2
        return err.T, p  # (100,1) (100,3)


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
