import numpy as np
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.RESNET = edict()
__C.RESNET.FIXED_BLOCKS = 1

__C.DATASETS = edict()
__C.DATASETS.ZEBRAFISH_IMG = edict()
__C.DATASETS.ZEBRAFISH_IMG.HOME = ""
__C.DATASETS.ZEBRAFISH = edict()
__C.DATASETS.ZEBRAFISH.HOME = ""

__C.VIS = edict()
__C.VIS.TSNE = edict()
__C.VIS.TSNE.SEQ = ["03"]
__C.VIS.TSNE.ID = "01"
__C.VIS.TSNE.GEN_GIF = False
__C.VIS.TSNE.GEN_CLIP = False
__C.VIS.TSNE.START = 100
__C.VIS.TSNE.END = 150
__C.VIS.TSNE.SEED = -1

__C.VIS.MODEL = edict()
__C.VIS.MODEL.BS = 128
__C.VIS.MODEL.BS_TRAIN = 8
__C.VIS.MODEL.DEAD_TYPE = 'mlp'
__C.VIS.MODEL.BACKBONE = 'cdc_resnet18'
__C.VIS.MODEL.FEAT_DIM = 128
__C.VIS.MODEL.SIZE = 32
__C.VIS.MODEL.EXP_NO = 'exp_xx'
__C.VIS.MODEL.RELOAD = ''
__C.VIS.MODEL.LR = 0.1
__C.VIS.MODEL.LR_DECAY = '30,70,80,95'
__C.VIS.MODEL.EVAL = ''
__C.VIS.MODEL.TSNE_SHOW = ''
__C.VIS.MODEL.START = 1
__C.VIS.MODEL.END = 1


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except Exception as e:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
