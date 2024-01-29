import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision import transforms

from modules.deep.model.net import get_model
from easydict import EasyDict as edict


class Structure:
    # Class variable that specifies expected fields
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # Set the arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        # Set the additional arguments (if any)
        extra_args = kwargs.keys() - self._fields
        for name in extra_args:
            setattr(self, name, kwargs.pop(name))

        if kwargs:
            raise TypeError('Duplicate values for {}'.format(','.join(kwargs)))

    def append_attr(self, name, value):
        if not hasattr(self, name):
            setattr(self, name, value)
        else:
            raise ValueError(f"{name} existed!")

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        doc = dict()
        for k in self._fields:
            doc[k] = getattr(self, k)
        return str(doc)


class ZebrafishDetSimple(Structure):
    _fields = ['frame',
               'c_x', 'c_y',  # head position
               'tl_x', 'tl_y', 'w', 'h',  # tlbr bbox
               'confidence',
               'app',
               'embedding']


class ZebrafishSequence:
    """
    detector: gt | yolo | pre_det
    """

    def __init__(self, detector=None,
                 det_path=None,
                 img_path=None,
                 gt_path=None,
                 model_path=None):

        self.detector = detector
        self.det_path = det_path
        self.gt_path = gt_path
        self.img_path = img_path

        self.top_det = []
        self.front_det = []

        self.top_dict = defaultdict(list)
        self.front_dict = defaultdict(list)

        self.top_frame = None
        self.front_frame = None

        self.param = {'cam1': {'head_diameter': 26,
                               'det_file': 'detections_2d_cam1.csv',
                               'img_file': 'imgT'},
                      'cam2': {'head_diameter': 50,
                               'det_file': 'detections_2d_cam2.csv',
                               'img_file': 'imgF'}}

        if self.detector == 'pre_det':
            self.load_data_dict()
        else:
            if model_path is not None:
                torch.manual_seed(42)
                torch.cuda.manual_seed(42)
                cudnn.benchmark = True
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                args = edict({'ckp': model_path, 'gpu_devices': None})
                self.model = get_model(args, device)
            else:
                self.model = None

            if self.detector == 'gt':
                self.load_gt_detection()
            elif self.detector == 'yolo':
                self.load_yolo_detection()

            if model_path is not None:
                import time
                start = time.time()
                self.gen_embedding()
                end = time.time() - start
                print("===END===: ", end)
                self.save_data_dict()

    def load_gt_detection(self):
        for view in self.param.keys():
            dets_df = pd.read_csv(Path(self.gt_path), sep=',', header=None)
            dets_df.columns = ['frame', 'id', '3d_x', '3d_y', '3d_z', 'camT_x', 'camT_y', 'camT_left', 'camT_top',
                               'camT_width', 'camT_height', 'camT_occlusion', 'camF_x', 'camF_y', 'camF_left',
                               'camF_top', 'camF_width', 'camF_height', 'camF_occlusion']
            if view == 'cam1':
                dets_df = dets_df[['frame', 'camT_x', 'camT_y', 'id']]
            else:
                dets_df = dets_df[['frame', 'camF_x', 'camF_y', 'id']]

            dets_df.columns = ['frame', 'cx', 'cy', 'id']
            images = sorted([str(im) for im in (Path(self.img_path) / self.param[view]['img_file']).rglob("*.jpg")])
            frames = dets_df['frame'].unique().astype(int)

            if view == 'cam1':
                self.top_frame = frames
            else:
                self.front_frame = frames

            for fr in frames:  # to do: speed up by multiple threads
                print(f'\r{fr}, {images[fr - 1]}', end='')
                self.load_det_per_frame(dets_df, fr, images[fr - 1], view)

    def load_yolo_detection(self):
        for view in self.param.keys():
            dets_df = pd.read_table(Path(self.det_path) / self.param[view]['det_file'], sep=',', header=None)
            dets_df.columns = ['frame', 'cx', 'cy', 'conf']
            images = sorted([str(im) for im in (Path(self.img_path) / self.param[view]['img_file']).rglob("*.jpg")])
            frames = dets_df['frame'].unique().astype(int)

            if view == 'cam1':
                self.top_frame = frames
            else:
                self.front_frame = frames

            for fr in frames:  # to do: speed up by multiple threads
                print(f'\r{fr}, {images[fr - 1]}', end='')
                self.load_det_per_frame(dets_df, fr, images[fr - 1], view)

    def load_det_per_frame(self, dets_df, fr, image, view):
        dets = dets_df[dets_df['frame'] == fr]
        img = cv2.imread(image)

        for d in dets.itertuples():
            frame = int(d.frame)
            tl_x = d.cx - self.param[view]['head_diameter'] / 2
            tl_y = d.cy - self.param[view]['head_diameter'] / 2
            w = self.param[view]['head_diameter']
            h = self.param[view]['head_diameter']

            patch = img[int(tl_y):int(tl_y + h), int(tl_x):int(tl_x + w)].astype(np.uint8)
            # print(fr, f"{sys.getsizeof(patch) / 1024:.1f} KB")
            feature = None

            element = ZebrafishDetSimple(
                frame, d.cx, d.cy,
                tl_x, tl_y, w, h,  # bbox
                1,  # conf
                # d.conf,  # conf
                patch,  # app
                feature  # embedding
            )
            if self.detector == 'gt':
                element.append_attr('id', d.id)

            if view == 'cam1':
                self.top_det.append(element)
            else:
                self.front_det.append(element)

    def gen_embedding(self):
        global patches
        for view in self.param.keys():
            if view == 'cam1':
                # embeddingNetTop
                net = self.model[0].module.embeddingNetA
                dets = self.top_det
            else:
                # embeddingNetFront
                net = self.model[0].module.embeddingNetB
                dets = self.front_det

            trans_func = transforms.ToTensor()
            batch_size = 1024
            result_embedding = []

            epochs = int(np.ceil(len(dets) / batch_size))
            indexes = [i for i in range(len(dets))]
            dets_index_batches = [indexes[i * batch_size: (i + 1) * batch_size] for i in range(epochs)]

            for dets_index in dets_index_batches:
                patches = []
                for _index in dets_index:
                    patch = cv2.resize(dets[_index].app, (64, 64))
                    patch = trans_func(patch).view((1, 3, 64, 64))
                    patches.append(patch)

                result_embedding.append(self.extract_embedding(patches, net))
                del patches

            torch.cuda.empty_cache()
            # np.vstack is time consume.
            # You can create a large np.zero matrix first if you have enough memory.
            res = np.vstack(result_embedding)
            print(f"{view} ", res.shape)

            for index in range(len(dets)):
                dets[index].embedding = res[index]

    def extract_embedding(self, frame_crops, net):
        with torch.no_grad():
            net.eval()
            bbox_crops = torch.cat(frame_crops)
            bbox_crops = bbox_crops.float().cuda()
            bbox_crops = Variable(bbox_crops)
            embedding = net(bbox_crops).detach().cpu().numpy()
            print("batch ", embedding.shape)
            return embedding

    def save_data_dict(self):
        data_dict = {
            'top': {
                'dets': self.top_det,
                'frames': self.top_frame
            },
            'front': {
                'dets': self.front_det,
                'frames': self.front_frame
            }
        }
        np.save(str(Path(self.det_path) / 'data_dict.npy'), data_dict)

    def load_data_dict(self):
        # embeddings have fix order, so don't change the detections file.
        data_dict = dict(np.load(str(Path(self.det_path) / 'data_dict.npy'), allow_pickle=True).item())
        self.top_det = data_dict['top']['dets']
        self.front_det = data_dict['front']['dets']

        self.top_frame = data_dict['top']['frames']
        self.front_frame = data_dict['front']['frames']

        for det in self.top_det:
            self.top_dict[det.frame].append(det)

        for det in self.front_det:
            self.front_dict[det.frame].append(det)


def main():
    index_dict = {
        'train': ['01', '02', '03', '04'],
        # 'train': ['03', '04'],
        # 'train': ['01'],
        'test': ['05', '06', '07', '08']
        # 'test': ['08']
    }
    # detections = r'gt_h\2d_detections'
    # detections = r'ddetr_h\2d_detections'
    # detections = r'yolo4_h_zy\2d_detections'
    detections = r'yolox_h_hc\2d_detections'
    # detections = r'yolox_h_sc\2d_detections'
    print(fr"We use the [{detections}] detections")
    for key in index_dict.keys():
        for index in index_dict[key]:
            det_dir = fr"D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\detections\{detections}\{index}"
            img_dir = fr'D:\Datasets\3DZeF20\{key}\ZebraFish-{index}'
            gt_dir = fr'D:\Datasets\3DZeF20\{key}\ZebraFish-{index}\gt\gt.txt'
            weight_dir = r'D:\Projects\FishTracking\for_release\ZebrafishTracking\modules\deep\weight' \
                         r'\exp_res_4data_03_cdc_resnet18_128_mlp_64_bs128_checkpoint_223.pth '
                         # r'\exp2-s4_(db1-0.1-r16)_cut([30, 65 ,100, 125][8][0.001]150)_se_dpb_cdc_resnet18_512_mlp_64_bs128_temp0.2_siamFalse_checkpoint_62.pth '
                         # ''
                         # r'\exp_24__temp0_2_resnet18_128_linear_64_bs32_checkpoint_54.pth '
            # weight_dir = None
            start = time.time()
            seq = ZebrafishSequence(detector='yolo',  # yolo | pre_det | gt
                                    det_path=det_dir,
                                    img_path=img_dir,
                                    gt_path=gt_dir,
                                    model_path=weight_dir)
            print(f"{index}-SEQ time cost: {time.time() - start: .3f} s")
            print(len(seq.top_det))
            print(len(seq.front_det))
            print(seq.front_det[0].embedding.shape)
            del seq


def evaluate():
    index_dict = {
        'train': ['03', '04'],
        # 'test': ['05', '06', '07', '08']
    }

    for key in index_dict.keys():
        for index in index_dict[key]:
            det_dir = fr"D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\detections\gt_h\{index}"
            img_dir = fr'D:\Datasets\3DZeF20\{key}\ZebraFish-{index}'
            gt_dir = fr'D:\Datasets\3DZeF20\{key}\ZebraFish-{index}\gt\gt.txt'
            weight_dir = r'D:\Projects\FishTracking\for_release\ZebrafishTracking\modules\deep\weight' \
                         r'\resnet18_o21_bs128_e12.pth '
            # weight_dir = None
            start = time.time()
            seq = ZebrafishSequence(detector='gt',  # yolo | pre_det | gt
                                    det_path=det_dir,
                                    img_path=img_dir,
                                    gt_path=gt_dir,
                                    model_path=weight_dir)
            print(f"{index}-SEQ time cost: {time.time() - start: .3f} s")
            print(len(seq.top_det))
            print(len(seq.front_det))
            print(seq.front_det[0].embedding.shape)
            del seq


# Example
if __name__ == '__main__':
    # evaluate()
    main()
