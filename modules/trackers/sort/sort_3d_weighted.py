from __future__ import print_function

import itertools

import cv2
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from filterpy.kalman import KalmanFilter

np.random.seed(42)


class Detection3D(object):
    def __init__(self, coordinates=None,
                 boxes_pair=([0, 0, 0, 0],
                             [0, 0, 0, 0]),
                 thetas=(None, None),
                 embeddings=(np.zeros((1, 512)), np.zeros((1, 512))),
                 frame=None,
                 velocity=0):
        self.coordinates = coordinates
        bboxA, bboxB = boxes_pair
        self.top_bbox = [bboxA[0], bboxA[1], bboxA[0] + bboxA[2], bboxA[1] + bboxA[3]]
        self.front_bbox = [bboxB[0], bboxB[1], bboxB[0] + bboxB[2], bboxB[1] + bboxB[3]]
        self.top_theta, self.front_theta = thetas
        self.top_embedding, self.front_embedding = embeddings
        self.frame = frame
        self.velocity = velocity


class Kalman3DPointTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, det: Detection3D):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array(
            [[1, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 1],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0]])

        self.kf.R *= 4.
        self.kf.P[3:, 3:] *= 10.  # give high uncertainty to the unobservable initial velocities
        self.kf.Q[3:, 3:] *= 0.11
        self.kf.x[:3] = det.coordinates.reshape((3, 1))
        self.time_since_update = 0
        self.id = Kalman3DPointTracker.count
        Kalman3DPointTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # here are 2d b-boxes
        self.frame = det.frame
        self.coordinates = det.coordinates
        self.top_bbox = det.top_bbox
        self.front_bbox = det.front_bbox
        self.top_theta = det.top_theta
        self.front_theta = det.front_theta

        self.top_embedding = det.top_embedding
        self.front_embedding = det.front_embedding

        # vel
        self.velocity = 0
        self.past_vel = []
        self.pos = self.coordinates

    def update_velocity(self, velocity):
        self.past_vel.append(np.linalg.norm(velocity))
        # if len(self.past_vel) >= 50:
        #     self.past_vel.pop(0)
        # mean_vel = np.average(self.past_vel, axis=0)
        mean_vel = np.median(self.past_vel, axis=0)
        self.velocity = mean_vel

    def get_vel(self, new_position):
        cur_vel = new_position - self.pos
        return cur_vel

    def update(self, det: Detection3D):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(det.coordinates[0:3].reshape((3, 1)))
        self.update_velocity(self.get_vel(self.get_state()))
        self.pos = self.get_state()

        # ...
        self.frame = det.frame
        self.coordinates = det.coordinates
        self.top_bbox = det.top_bbox
        self.front_bbox = det.front_bbox
        self.top_theta = det.top_theta
        self.front_theta = det.front_theta

        self.top_embedding = det.top_embedding
        self.front_embedding = det.front_embedding

    def predict(self):
        """
        Advances the state vector and returns the predicted 3d point estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[0:3].reshape((1, 3))


def linear_assignment(cost_matrix):
    # print(cost_matrix)
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


def limit_theta(distTheta):
    if distTheta < -90:
        distTheta = distTheta + 180
    elif distTheta > 90:
        distTheta = distTheta - 180
    return distTheta / 90


def iou_scoring(boxA, boxB):
    #
    w1 = boxA[2] - boxA[0]
    h1 = boxA[3] - boxA[1]
    w2 = boxB[2] - boxB[0]
    h2 = boxB[3] - boxB[1]

    boxA_area = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    center_x1 = (boxA[0] + boxA[2]) / 2
    center_y1 = (boxA[1] + boxA[3]) / 2
    center_x2 = (boxB[0] + boxB[2]) / 2
    center_y2 = (boxB[1] + boxB[3]) / 2

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    c_l = min(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = max(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = min(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = max(center_y1 + h1 / 2, center_y2 + h2 / 2)

    x1C = min(boxA[0], boxB[0])
    y1C = min(boxA[1], boxB[1])
    x2C = max(boxA[2], boxB[2])
    y2C = max(boxA[3], boxB[3])

    # calc area of Bc
    area_c = (x2C - x1C) * (y2C - y1C)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    c_diag = np.clip((c_r - c_l), a_max=np.inf, a_min=0) ** 2 + np.clip((c_b - c_t), a_max=np.inf, a_min=0) ** 2

    union = boxA_area + boxB_area - inter_area
    u = inter_diag / c_diag
    iou = inter_area / float(union)

    dious = iou - u
    giou = iou - (area_c - union) / area_c

    iou = np.clip(iou, a_max=1, a_min=0)
    giou = np.clip(giou, a_max=1, a_min=-1)
    diou = np.clip(dious, a_max=1, a_min=-1)

    return iou, (giou + 1) / 2, (diou + 1) / 2


def cost_calc(det_current, det_tracker, app_item_weight=0.2, mean=None, cov=None):
    def calc_embedding_distance(e1, e2, size=512):
        e1, e2 = np.array(e1), np.array(e2)
        e1, e2 = torch.from_numpy(e1), torch.from_numpy(e2)
        e1, e2 = e1.view(-1, size), e2.view(-1, size)
        return 1 - (F.cosine_similarity(e1, e2).numpy() + 1) / 2

    def euclidean_distances(current_point, track_point):
        return np.array([np.linalg.norm(p[0] - p[1]) for p in zip(current_point, track_point)])

    det_tracker = [e for e in det_tracker]
    det_current = [e for e in det_current]

    res = list(itertools.product(det_current, det_tracker))
    delta_frames = []

    current_coordinates = []
    tracker_coordinates = []

    current_top_embeddings = []
    current_front_embeddings = []
    current_iou_matrix = []

    tracker_top_embeddings = []
    tracker_front_embeddings = []
    tracker_iou_matrix = []

    current_velocity = []
    tracker_velocity = []

    n, m = len(det_current), len(det_tracker)

    for pair in res:
        current: Detection3D = pair[0]
        tracker: Detection3D = pair[1]

        delta_frames.append(abs(current.frame - tracker.frame))

        current_coordinates.append(np.array(current.coordinates))
        tracker_coordinates.append(np.array(tracker.coordinates))
        current_velocity.append(np.array(current.velocity))
        tracker_velocity.append(np.array(tracker.velocity))

        current_top_embeddings.append(current.top_embedding)
        current_front_embeddings.append(current.front_embedding)
        tracker_top_embeddings.append(tracker.top_embedding)
        tracker_front_embeddings.append(tracker.front_embedding)

        _, _, diou = iou_scoring(current.top_bbox, tracker.top_bbox)
        current_iou_matrix.append(diou)
        _, _, diou = iou_scoring(current.front_bbox, tracker.front_bbox)
        tracker_iou_matrix.append(diou)

    euclidean_dist_matrix = euclidean_distances(current_coordinates, tracker_coordinates)
    euclidean_dist_matrix = 1 - np.exp(-euclidean_dist_matrix / 5)
    velocity_dist_matrix = np.abs(np.array(current_velocity) - np.array(tracker_velocity))
    # current_iou_matrix = np.array(current_iou_matrix)
    # tracker_iou_matrix = np.array(tracker_iou_matrix)

    # calc delta_frames  np.pow(decay, delta_frames)  ==> decay = 0.8
    app_simt_matrix = calc_embedding_distance(current_top_embeddings, tracker_top_embeddings,
                                              size=tracker_top_embeddings[0].shape[0])
    app_simf_matrix = calc_embedding_distance(current_front_embeddings, tracker_front_embeddings,
                                              size=tracker_top_embeddings[0].shape[0])

    mask = np.where((app_simt_matrix - app_simf_matrix) < 0, 1, 0)
    app_cost_matrix = app_simt_matrix * mask + app_simf_matrix * (1 - mask)
    app_item_weight = np.multiply(app_item_weight, np.asarray([pow(0.9, d-1) for d in delta_frames]))
    # print(delta_frames)

    iou_cost_matrix = (1 - app_item_weight) * euclidean_dist_matrix + app_item_weight * app_cost_matrix
    # print(app_item_weight)

    return euclidean_dist_matrix.reshape(n, m), \
        velocity_dist_matrix.reshape(n, m), \
        app_cost_matrix.reshape(n, m), \
        app_simt_matrix.reshape(n, m), \
        app_simf_matrix.reshape(n, m), \
        iou_cost_matrix.reshape(n, m)


class Sort3DWeighted(object):
    def __init__(self, max_age=1, min_hits=3, cost_threshold=5, app_item_weight=0.2):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.cost_threshold = cost_threshold
        self.app_item_weight = app_item_weight
        self.trackers = []
        self.frame_count = 0

        self.app_profile = []
        self.euc_profile = []
        self.fusion_profile = []

    def associate_detections_to_trackers(self, detections, trackers, cost_threshold=5, app_item_weight=0.2,
                                         mean=None, cov=None, images_path=None):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 3), dtype=int)

        eu_cost_matrix, vel_cost_matrix, \
            app_cost_matrix, \
            appt_cost_matrix, \
            appf_cost_matrix, \
            iou_cost_matrix = cost_calc(detections, trackers,
                                        app_item_weight=app_item_weight,
                                        mean=mean, cov=cov)

        self.app_profile.append(app_cost_matrix)
        self.euc_profile.append(eu_cost_matrix)

        # Eliminate objects that exceed the threshold
        cost_matrix = (1 - app_item_weight) * eu_cost_matrix + app_item_weight * app_cost_matrix
        self.fusion_profile.append(iou_cost_matrix)
        self.fusion_profile.append(cost_matrix)

        matched_indices = linear_assignment(cost_matrix)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with high dist
        matches = []
        for m in matched_indices:
            if cost_matrix[m[0], m[1]] > cost_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, dets):

        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = [Detection3D() for _ in range(len(self.trackers))]
        ret = []
        # to_del = []
        for t, trk in enumerate(trks):
            kf: Kalman3DPointTracker = self.trackers[t]
            pos = kf.predict()[0]
            trk.coordinates = [pos[0], pos[1], pos[2]]
            # trk.coordinates = kf.coordinates
            kf.update_velocity(kf.get_vel(trk.coordinates[0]))
            trk.velocity = kf.velocity
            trk.frame = kf.frame

            trk.top_bbox, trk.front_bbox = kf.top_bbox, kf.front_bbox
            trk.top_theta, trk.front_theta = kf.top_theta, kf.front_theta
            trk.top_embedding, trk.front_embedding = kf.top_embedding, kf.front_embedding

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks,
                                                                                        self.cost_threshold,
                                                                                        self.app_item_weight)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = Kalman3DPointTracker(dets[i])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and \
                    (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((trk.coordinates, [trk.id + 1], trk.top_bbox, trk.front_bbox)).reshape(1, -1))
                # +1 as MOT benchmark requires positive

            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 3))
