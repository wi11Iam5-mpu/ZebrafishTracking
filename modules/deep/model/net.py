import os
import torch
import torch.nn as nn

from .resnet_big_se import EmbeddingResnet as EmbeddingResnetBIG


def mask_tensor(_img, _mask):
    shape = _img[:1].shape
    batch_mask = torch.cat([torch.ones(shape).bool().cuda() * i for i in _mask], dim=0)
    _masked_output = _img * batch_mask.int().float()
    return _masked_output


class PseudoTripletNet(nn.Module):
    def __init__(self, embeddingNetA, embeddingNetB):
        super(PseudoTripletNet, self).__init__()
        self.embeddingNetA = embeddingNetA
        self.embeddingNetB = embeddingNetB

    def forward(self, batch):
        """
        0 means top view => embeddingNetA
        1 means front view => embeddingNetB
        samples: anchor_v1_img, anchor_v2_img,
                 anchor_v1_img_aug, anchor_v2_img_aug,
                 pos_v1_img, pos_v2_img,
                 pos_i_v1_img, pos_i_v2_img,
                 neg_v1_imgs, neg_v2_imgs, fake_label
        """
        # samples[0] N*C*W*H samples[0] N*[views]
        _samples, _masks = batch
        result = []
        for i, mask in enumerate(_masks):
            mask_a = [1 - i for i in mask]
            mask_b = mask
            sample = _samples[:, i, :, :, :]
            res = mask_tensor(self.embeddingNetA(sample), mask_a) + \
                  mask_tensor(self.embeddingNetB(sample), mask_b)
            result.append(res)
        return result


def get_model(args, device):
    from configs.base_config import cfg, cfg_from_file
    cfg_from_file(r'D:\Projects\FishTracking\for_release\ZebrafishTracking\configs\test.yaml')
    net_A, net_B = EmbeddingResnetBIG(name=cfg.VIS.MODEL.BACKBONE,
                                      head=cfg.VIS.MODEL.DEAD_TYPE,
                                      feat_dim=cfg.VIS.MODEL.FEAT_DIM), \
                   EmbeddingResnetBIG(name=cfg.VIS.MODEL.BACKBONE,
                                      head=cfg.VIS.MODEL.DEAD_TYPE,
                                      feat_dim=cfg.VIS.MODEL.FEAT_DIM)
    model = PseudoTripletNet(net_A, net_B)

    start_epoch = 1
    # Load weights if provided
    if args.ckp:
        if os.path.isfile(args.ckp):
            try:
                model_dict = torch.load(args.ckp)['state_dict']
            except Exception:
                model_dict = torch.load(args.ckp, map_location='cpu')['state_dict']

            print("=> Loaded checkpoint '{}'".format(args.ckp))

            model_dict_modA = {}
            model_dict_modB = {}

            for key, value in model_dict.items():
                if "embeddingNetA" in key:
                    new_key = '.'.join(key.split('.')[2:])
                    model_dict_modA[new_key] = value
                elif "embeddingNetB" in key:
                    new_key = '.'.join(key.split('.')[2:])
                    model_dict_modB[new_key] = value

            model.embeddingNetA.load_state_dict(model_dict_modA)
            model.embeddingNetB.load_state_dict(model_dict_modB)
            start_epoch = torch.load(args.ckp)['epoch']

            print("=> Loaded checkpoint '{}'".format(args.ckp))
        else:
            print("=> No checkpoint found at '{}'".format(args.ckp))

    model = nn.DataParallel(model, device_ids=args.gpu_devices)
    model = model.to(device)

    return model, start_epoch
