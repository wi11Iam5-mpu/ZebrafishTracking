import math
import time

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv

import pandas as pd


class CameraParams(object):

    def __init__(self, k, dist, r, t, extr_mat, intr_mat, pos, plane_normal, plane_points):
        self.k = k
        self.dist = dist
        self.r = r
        self.t = t
        self.extr_mat = extr_mat
        self.intr_mat = intr_mat
        self.pos = pos
        self.plane_normal = plane_normal
        self.plane_points = plane_points


def get_cams_params():
    cam_top = CameraParams(k=np.array([[1.4907767e+03, 0.0000000e+00, 1.3438746e+03],
                                       [0.0000000e+00, 1.4636407e+03, 7.8186383e+02],
                                       [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]),
                           dist=np.array([-7.8173008e+00, 2.7467192e+01, 3.1850096e-03, -1.1391344e-03,
                                          8.0843201e+00, -7.8022938e+00, 2.7339403e+01, 8.6123762e+00,
                                          0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                          0.0000000e+00, 0.0000000e+00]),
                           r=np.array([[-0.04449836],
                                       [0.01277325],
                                       [0.01699574]]),
                           t=np.array([[-15.060983],
                                       [-14.814506],
                                       [33.366383]]),
                           extr_mat=np.array(
                               [[9.9977404e-01, -1.7272992e-02, 1.2390012e-02, -1.5060983e+01],
                                [1.6704720e-02, 9.9886572e-01, 4.4588853e-02, -1.4814506e+01],
                                [-1.3146142e-02, -4.4371806e-02, 9.9892861e-01, 3.3366383e+01]]),
                           intr_mat=np.array([[1.4907767e+03, 0.0000000e+00, 1.3438746e+03],
                                              [0.0000000e+00, 1.4636407e+03, 7.8186383e+02],
                                              [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]),
                           pos=np.array([[15.7436905, 16.01808, -32.483463]]),
                           plane_normal=np.array([-0., -0., -1.]),
                           plane_points=np.array([[0., 0., 0.],
                                                  [29., 0., 0.],
                                                  [29., 29., 0.],
                                                  [0., 29., 0.]]))
    cam_side = CameraParams(k=np.array([[1.4604250e+03, 0.0000000e+00, 1.3482039e+03],
                                        [0.0000000e+00, 1.4578763e+03, 7.7426624e+02],
                                        [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]),
                            dist=np.array([-6.2602341e-02, -1.4225453e+00, 2.2716711e-03, -1.4766378e-04,
                                           -4.5899123e-01, -7.3587060e-02, -1.4103940e+00, -4.5532566e-01,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00]),
                            r=np.array([[-1.3420784],
                                        [-0.00625672],
                                        [0.01343344]]),
                            t=np.array([[-14.445],
                                        [-11.662717],
                                        [47.74172]]),
                            extr_mat=np.array(
                                [[9.9990571e-01, -6.1434601e-03, -1.2280220e-02, -1.4445000e+01],
                                 [1.3353245e-02, 2.2666611e-01, 9.7388101e-01, -1.1662717e+01],
                                 [-3.1994891e-03, -9.7395313e-01, 2.2672677e-01, 4.7741718e+01]]),
                            intr_mat=np.array([[1.4604250e+03, 0.0000000e+00, 1.3482039e+03],
                                               [0.0000000e+00, 1.4578763e+03, 7.7426624e+02],
                                               [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]]),
                            pos=np.array([[14.752122, 49.052998, 0.3563851]]),
                            plane_normal=np.array([0., 1., 0.]),
                            plane_points=np.array([[0., 29., 0.],
                                                   [29., 29., 0.],
                                                   [29., 29., 15.],
                                                   [0., 29., 15.]]))
    return cam_top, cam_side


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


def back_project_points():
    ...


def forward_project_points(points, cam_top_param, cam_side_param):
    refPoints_top = get_ref_point_batch(points, cam_top_param.pos, cam_top_param.plane_normal,
                                        cam_top_param.plane_points)
    refPoints_side = get_ref_point_batch(points, cam_side_param.pos, cam_side_param.plane_normal,
                                         cam_side_param.plane_points)

    points_2d_ref_top = project_points(refPoints_top, cam_top_param.extr_mat, cam_top_param.intr_mat,
                                       cam_top_param.dist)
    points_2d_ref_side = project_points(refPoints_side, cam_side_param.extr_mat, cam_side_param.intr_mat,
                                        cam_side_param.dist)
    return points_2d_ref_top, points_2d_ref_side


def test_world2cam():
    x = np.linspace(0, 30 - 1, 10)
    y = np.linspace(0, 30 - 1, 10)
    z = np.linspace(0, 15 - 1, 5)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    cam_top_param, cam_side_param = get_cams_params()
    start = time.time()

    # refPoints_top = get_ref_point_batch(points, cam_top_param.pos, cam_top_param.plane_normal,
    #                                     cam_top_param.plane_points)
    # refPoints_side = get_ref_point_batch(points, cam_side_param.pos, cam_side_param.plane_normal,
    #                                      cam_side_param.plane_points)
    #
    # points_2d_ref_top = project_points(refPoints_top, cam_top_param.extr_mat, cam_top_param.intr_mat,
    #                                    cam_top_param.dist)
    # points_2d_ref_side = project_points(refPoints_side, cam_side_param.extr_mat, cam_side_param.intr_mat,
    #                                     cam_side_param.dist)

    points_2d_ref_top, points_2d_ref_side = forward_project_points(points, cam_top_param, cam_side_param)

    end = time.time()
    print("Cost time: ", end - start)

    show_ref_points(points_2d_ref_top)
    show_ref_points(points_2d_ref_side)


def show_ref_points(points_2d_ref):
    img = np.zeros((1520, 2704, 3), np.float32)
    for point in points_2d_ref:
        # for point, color in points:
        cv2.circle(img, point[:2], 8, (0, 233, 0), -1)
    # 显示图像
    height, width = img.shape[0:2]
    max_size = 800, 600  # width, height
    if width > max_size[0] or height > max_size[1]:
        scale = min(max_size[0] / width, max_size[1] / height)
        width, height = int(width * scale), int(height * scale)
        img = cv2.resize(img, (width, height))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class AnimatedVicsekModel(object):
    """An animated scatter plot of Vicsek Model using matplotlib.animations.FuncAnimation."""

    def __init__(self, numpoints=50, file_name='ZebraFishSim-03.txt', total_cnt=1000, interpolate_n=2):
        self.numpoints = numpoints  # the number of particles
        self.Lx, self.Ly, self.Lz = 30, 30, 30  # Space x y z
        self.R = 10  # field-of-view radius
        self.Step = 1.5  # step
        self.alpha = 0.6  # speed factor
        self.beta = 1 - self.alpha  # inertial factor
        self.gamma = 0.5  # attenuation factor
        self.eta = 0.5  # noise intensity
        self.T = 10  # simulation time
        self.K = 5  # Number of neighbors affected
        self.vis_theta = math.pi  # viewing angle 60°

        self.stream = self.velocity_update()

        self.cam_top_param, self.cam_side_param = get_cams_params()

        # Setup the figure and axes...
        self.fig = plt.figure(figsize=(12, 4))
        self.ax = self.fig.add_subplot(131, projection='3d')  # Create a 3D coordinate axis
        self.ax_top = self.fig.add_subplot(132)
        self.ax_side = self.fig.add_subplot(133)
        self.timetext = None
        self.scat = None  #
        self.quiv = None  #
        self.scat_top = None
        self.scat_side = None
        # Then setup FuncAnimation.
        # self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.T, interval=50,
        #                                    init_func=self.setup_plot)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=50, init_func=self.setup_plot)

        self.output_name = file_name
        self.output_limit = total_cnt
        self.interpolate_n = interpolate_n

    @staticmethod
    def distance(x1, x2):
        """Calculate the distance of two particles."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    @staticmethod
    def angle(v1, v2):
        """Calculate the ange of two particles."""
        return math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def apply_bc(self, x, v):
        """处理边界条件，如果超出边界则反弹回来. 边界保护 margin
            返回调整后的位置和速度
        """
        margin = 0.5
        bc_low = (0 + margin, 0 + margin, 0 + margin)  # 低边界条件
        bc_high = (self.Lx - margin, self.Ly - margin, (self.Lz - margin) // 2)  # 高边界条件

        for i in range(3):  # 遍历三个坐标轴
            mask1 = x[:, i] < bc_low[i]  # 找到小于bc_low的位置
            mask2 = x[:, i] > bc_high[i]  # 找到大于bc_high的位置
            x[mask1, i] = -x[mask1, i]  # 将小于bc_low的位置反转为正数
            x[mask2, i] = 2 * bc_high[i] - x[mask2, i]  # 将大于bc_high的位置反转为小于bc_high的数
            v[mask1, i] = -v[mask1, i]  # 将小于bc_low的位置对应的速度反向
            v[mask2, i] = -v[mask2, i]  # 将大于bc_high的位置对应的速度反向
        return x, v

    def find_neighbors(self, x, v, i):
        neighbors = []  # 存储邻域内的邻居粒子的索引, 初始化邻居集合为空列表
        for j in range(self.numpoints):  # 对每个粒子 j
            if i != j and self.distance(x[i], x[j]) < self.R:  # 排除自身, 如果 j 在 i 的视野内
                if self.angle(v[i], v[j]) < self.vis_theta:  # 引入可视角度
                    neighbors.append(j)  # 将 j 加入邻域列表

        if len(neighbors) > self.K:  # 如果邻居数量大于拓扑距离限制，就只保留最近的k个邻居
            dists = [self.distance(x[i], x[j]) for j in neighbors]  # 计算所有邻居与粒子i的距离
            sorted_indices = np.argsort(dists)  # 对距离进行排序，得到排序后的索引
            neighbors = [neighbors[sorted_indices[i]] for i in range(self.K)]  # 根据索引保留最近的k个邻居

        return neighbors

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        xyz, v, s, c = next(self.stream)
        self.scat = self.ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='b', s=s)
        self.quiv = self.ax.quiver(xyz[:, 0], xyz[:, 1], xyz[:, 2], self.Step * v[:, 0], self.Step * v[:, 1],
                                   self.Step * v[:, 2], length=1.5, color='r')
        self.ax.set_xlim(0, self.Lx)
        self.ax.set_ylim(0, self.Ly)
        self.ax.set_zlim(0, self.Lz)

        self.scat_top = self.ax_top.scatter(xyz[:, 0], xyz[:, 1], c='b', s=s)
        self.scat_side = self.ax_side.scatter(xyz[:, 1], xyz[:, 2], c='b', s=s)

        # self.ax_top.set_xlim(0, self.Lx)
        # self.ax_side.set_xlim(0, self.Lx)
        # self.ax_top.set_ylim(0, self.Ly)
        # self.ax_side.set_ylim(0, self.Ly)
        self.ax_top.set_xlim(0, 2704)
        self.ax_side.set_xlim(0, 2704)
        self.ax_top.set_ylim(0, 1520)
        self.ax_side.set_ylim(0, 1520)

        self.timetext = self.ax.text(20, 0, 40, '')

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.quiv, self.scat_top, self.scat_side

    def velocity_update(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        # 初始化粒子的位置和速度 斑马鱼运动能力在 10-20 cm/s
        xyz = np.random.uniform(0, self.Lx, size=(self.numpoints, 3))  # 随机分布在 [0, Lx] x [0, Ly] x [0, Lz] 的立方体内
        v = np.random.uniform(-1, 1, size=(self.numpoints, 3))  # 随机分布在 [-1, 1] x [-1, 1] x [-1, 1] 的速度空间内
        v = v / np.linalg.norm(v, axis=1, keepdims=True)  # 归一化速度大小为 1
        s, c = (10, 2)
        tick = 0
        while True:
            tick = tick + 1
            for i in range(self.numpoints):  # 对每个粒子 i
                neighbors = self.find_neighbors(xyz, v, i)
                if len(neighbors) > 0:  # 如果 i 有邻居
                    V_avg = np.mean(v[neighbors], axis=0)  # 计算邻居的平均速度方向
                    xi = np.random.uniform(-self.eta / 2, self.eta / 2, size=3)  # 生成一个随机噪声矢量
                    if np.random.rand() < 0.2:
                        F = v[i] + xi  # 有一定概率称为领导者，采用领导者速度
                    else:
                        # F = self.alpha * (V_avg - v[i]) + self.beta * v[i] + xi  # 计算合力矢量
                        F = self.alpha * V_avg + self.beta * v[i] + xi  # 计算合力矢量

                    if self.angle(v[i], F) > self.vis_theta:
                        v[i] = v[i] + F  # 更新速度方向
                    else:
                        v[i] = self.gamma * (v[i] + F)  # 更新速度方向
                        # 更新速度大小（考虑衰减）
                        # v = v0 * np.exp(-k * t)

                    v[i] = v[i] / np.linalg.norm(v[i])  # 归一化速度大小为 1
                xyz[i] = xyz[i] + self.Step * v[i]  # 更新位置
            xyz, v = self.apply_bc(xyz, v)

            yield xyz, v, s, c

    def save_data(self, data, filename, counter, limit):
        """
        export data format:
        frame id 3d_x 3d_y 3d_z
        camT_x camT_y camT_left camT_top camT_width camT_height camT_occlusion
        camF_x camF_y camF_left camF_top camF_width camF_height camF_occlusion
        """
        data_2d_top, data_2d_side = forward_project_points(data[:, 0:3], self.cam_top_param, self.cam_side_param)

        data_3d_x, data_3d_y, data_3d_z = data[:, 0], data[:, 1], data[:, 2]
        # camT_x, camT_y = list(data_3d_x), list(data_3d_y)
        camT_x, camT_y = data_2d_top[:, 0], data_2d_top[:, 1]
        camT_width, camT_height = [30], [30]
        camT_left, camT_top = list(camT_x + 0.5 * camT_width[0]), list(camT_y + 0.5 * camT_height[0])
        camT_occlusion = [0]

        # camF_x, camF_y = list(data_3d_x), list(data_3d_z)
        camF_x, camF_y = data_2d_side[:, 0], data_2d_side[:, 1]
        camF_width, camF_height = [50], [50]
        camF_left, camF_top = list(camF_x + 0.5 * camF_width[0]), list(camF_y + 0.5 * camF_height[0])
        camF_occlusion = [0]

        frame = [int(counter + 1)]
        ids = [int(i + 1) for i in range(len(data_3d_x))]

        data_line = [frame, ids, data_3d_x, data_3d_y, data_3d_z,
                     camT_x, camT_y, camT_left, camT_top, camT_width, camT_height, camT_occlusion,
                     camF_x, camF_y, camF_left, camF_top, camF_width, camF_height, camF_occlusion]

        # 处理不同长度的项
        # 找出最长的变量的长度
        max_len = max(len(x) for x in data_line)

        # 定义一个空的二维数组
        arr = np.empty((0, max_len), dtype=float)

        # 遍历列表中的每个变量
        for x in data_line:
            # 使用pad函数对变量进行填充，使其长度等于最长的变量的长度
            # 填充的方式为'wrap'，表示重复原始数组的元素
            # 填充的值为None，表示不指定特定的值
            padded_x = np.pad(x, (0, max_len - len(x)), mode='wrap')
            # 把填充后的变量添加到二维数组中
            arr = np.append(arr, [padded_x], axis=0)

        # 如果计数器小于预设值，就把数据追加到文件中
        if counter < limit:
            # 打开文件，设置模式为'a'，表示追加
            with open(f'{filename}.txt', 'a', newline='') as f:
                # 创建一个csv写入对象
                writer = csv.writer(f)
                # 写入一行数据
                # writer.writerow(arr)
                writer.writerows(arr.T)

        # 如果计数器等于预设值，就结束保存
        else:
            # 打印提示信息
            print('已达到预设值，停止保存')
            self.interpolate(n=self.interpolate_n)
            exit()

    def update(self, iteration):
        """Update the scatter plot."""
        xyz, v, s, c = next(self.stream)
        data_2d_top, data_2d_side = forward_project_points(xyz[:, 0:3], self.cam_top_param, self.cam_side_param)
        # Set x and y data...
        self.scat._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        # self.scat_top._offsets = np.c_[xyz[:, 0], xyz[:, 1]]
        # self.scat_side._offsets = np.c_[xyz[:, 1], xyz[:, 2]]
        self.scat_top._offsets = data_2d_top
        self.scat_side._offsets = data_2d_side
        # Set arrow data...
        self.quiv.set_segments(np.array([xyz, xyz + self.Step * v]).transpose(1, 0, 2))  # 更新箭头图的数据
        # self.ax.view_init(elev=15, azim=i / 5)  # 更新视角

        self.timetext.set_text(f'Frame {iteration}')

        # 保存数据
        self.save_data(xyz, self.output_name, counter=iteration, limit=self.output_limit)

        return self.scat, self.quiv, self.scat_top, self.scat_side

    def interpolate(self, n=2):
        # sep参数指定分隔符
        df = pd.read_csv(
            fr"{self.output_name}.txt",
            sep=',', header=None)

        df.columns = ['frame', 'id', 'data_3d_x', 'data_3d_y', 'data_3d_z',
                      'camT_x', 'camT_y', 'camT_left', 'camT_top', 'camT_width', 'camT_height', 'camT_occlusion',
                      'camF_x', 'camF_y', 'camF_left', 'camF_top', 'camF_width', 'camF_height', 'camF_occlusion']

        print(df.shape)

        # 定义一个函数，用于在每个帧之间插入n个空白行
        def insert_blank_rows(df, n):
            # 创建一个空的DataFrame
            new_df = pd.DataFrame()
            # 遍历原始DataFrame的每一行
            for i, row in df.iterrows():
                # 将当前行添加到新的DataFrame中
                new_df = new_df.append(row)
                # 如果当前行不是最后一行
                if i < len(df) - 1:
                    # 获取下一行的帧号
                    next_frame = df.loc[i + 1, "frame"]
                    # 计算需要插入的空白行的帧号
                    blank_frames = [row["frame"] + (j + 1) * (next_frame - row["frame"]) / (n + 1) for j in range(n)]
                    # 创建一个空白行的DataFrame，用NaN填充除了帧号和id号以外的列
                    blank_df = pd.DataFrame({"frame": blank_frames, "id": row["id"]})
                    blank_df[['frame', 'data_3d_x', 'data_3d_y', 'data_3d_z',
                              'camT_x', 'camT_y',
                              'camF_x', 'camF_y', ]] = np.nan
                    # 将空白行的DataFrame添加到新的DataFrame中
                    new_df = new_df.append(blank_df)
            # 返回新的DataFrame
            return new_df

        # 调用函数，插入n个空白行
        df = insert_blank_rows(df, n)

        # 对每个id号分组
        groups = df.groupby("id")

        # 定义一个空的DataFrame来存储结果
        result = pd.DataFrame()

        # 对每个分组进行插值
        for name, group in groups:
            # # 设置插值数为2
            # n = 2

            # 对x,y,z,cx,cy列进行线性插值
            group[['frame', 'data_3d_x', 'data_3d_y', 'data_3d_z', 'camT_x', 'camT_y',
                   'camF_x', 'camF_y', ]] = group[
                ['frame', 'data_3d_x', 'data_3d_y', 'data_3d_z', 'camT_x', 'camT_y',
                 'camF_x', 'camF_y', ]].interpolate(
                method="linear", limit=n,
                limit_direction="both")

            # 将插值后的分组添加到结果中
            group['frame'] = [int(i + 1) for i in range(group.shape[0])]
            result = result.append(group)

        pd.set_option('expand_frame_repr', False)

        # 打印结果
        df_result = result.sort_values(by=["frame", "id"], ascending=True).reset_index(drop=True)
        print(df_result)
        print(df_result.shape)
        df_result.to_csv(f'{self.output_name}.csv', header=None, index=None)


if __name__ == '__main__':
    points_num = 50
    frame_cnt = 100
    exp = 100

    a = AnimatedVicsekModel(numpoints=points_num, file_name=f'ZebraFishSim-{points_num}-{frame_cnt}-{exp}',
                            total_cnt=frame_cnt, interpolate_n=10)

    if False:
        a.ani.save("Z1.mp4", fps=60, writer='ffmpeg')
    else:
        plt.show()

    # test_world2cam()
