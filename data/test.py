# import pickle

# data = open('/remote-home/share/SHTperson/save_info/sht_infos_train23568.pkl', 'rb')
# train_info = pickle.load(data)
# len(train_info)
# 5279

# train_info[0].keys() 
# dict_keys(['image', 'point_cloud', 'annos'])
# train_info[0]['image']
# {'image_idx': 0, 'image_path': '/remote-home/share/SHTperson/4-20/1/left/1_1618891263.01540.jpg'}
# train_info[0]['point_cloud']
# {'num_features': 4, 'point_cloud_path': '/remote-home/share/SHTperson/4-20/1/pcd/1618891263.764833792.bin'}
# train_info[0]['annos'].keys()
# dict_keys(['position', 'dimensions', 'occlusion', 'rotation', 'image_bbox'])

# len(train_info[0]['annos']['position'])
# 14
# len(train_info[0]['annos']['position'][0])
# 3
# len(train_info[0]['annos']['dimensions'])
# 14
# len(train_info[0]['annos']['dimensions'][0])
# 3
# len(train_info[0]['annos']['occlusion'])
# 14
# train_info[0]['annos']['occlusion'][0]
# int
# len(train_info[0]['annos']['rotation'])
# 14
# train_info[0]['annos']['rotation'][0]
# float
# len(train_info[0]['annos']['image_bbox'])
# 10 ?????
# len(train_info[0]['annos']['image_bbox'][0])
# 8

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import mmcv
import torch

class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results



config_file = '/root/PointPainting/painting/mmseg/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py'
checkpoint_file = '/root/PointPainting/painting/mmseg/checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
image_file = '/remote-home/share/SHTperson/3-22/1/left/1_1616406608.97715.jpg'
model = init_segmentor(config_file, checkpoint_file,
                            device='cuda:2')  # TODO edit here if you want to use different device

cfg = model.cfg
device = next(model.parameters()).device  # model device
# build the data pipeline
test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
test_pipeline = Compose(test_pipeline)
# prepare data
data = dict(img=image_file)
data = test_pipeline(data)
data = collate([data], samples_per_gpu=1)
if next(model.parameters()).is_cuda:
    # scatter to specified GPU
    data = scatter(data, [device])[0]
else:
    data['img_metas'] = [i.data[0] for i in data['img_metas']]

# forward the model
with torch.no_grad():
    a = model.whole_inference(data['img'][0], data['img_metas'][0], True)  # [1, 19, 720, 1280]
    b = model.encode_decode(data['img'][0], data['img_metas'])  # [1, 19, 1024, 1820]
    c = model.extract_feat(data['img'][0])
    # tuple, c[0]:[1, 256, 256, 455]; c[1]:[1, 512, 128, 228]; c[2]:[1, 1024, 128, 228] c[3]:[1, 2048, 128, 228]
    d = model._decode_head_forward_test(c, data['img_metas'])  # [1, 19, 256, 455]
    e = model.inference(data['img'][0], data['img_metas'][0], True)  # [1, 19, 720, 1280] softmax
    result = model(return_loss=False, rescale=True, **data)
import pdb
pdb.set_trace()

