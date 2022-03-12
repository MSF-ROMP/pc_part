from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import mmcv
import torch
import time
import numpy as np
import os

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


def generate(folder, bin_folder, bin_painting_folder):
    file_list = os.listdir(bin_folder)
    image_path = os.path.join(folder, 'left')
    image_file_list = os.listdir(image_path)
    file_list.sort()
    image_file_list.sort()
    assert len(file_list) == len(image_file_list)
    if os.path.exists(bin_painting_folder):
        pass
    else:
        os.makedirs(bin_painting_folder)

    for i in range(len(file_list)):
        (filename, extension) = os.path.splitext(file_list[i])
        bin_file = os.path.join(bin_folder, filename) + '.bin'
        bin_file_new = os.path.join(bin_painting_folder, filename) + '.bin'
        if os.path.exists(bin_file_new):
            points_new = np.fromfile(bin_file_new, dtype=np.float32)
            # print(points_new.shape[0])
            if points_new.shape[0] == 6029312:
                continue
        # import pdb
        # pdb.set_trace()
        image_file = os.path.join(image_path, image_file_list[i])
        # # single frame test
        # bin_file = '/remote-home/share/SHTperson/3-22/1/bin_v1/1616406608.796536064.bin'
        # bin_file_new = '/remote-home/share/SHTperson/3-22/1/bin_painting/1616406608.796536064.bin'
        # image_file = '/remote-home/share/SHTperson/3-22/1/left/1_1616406608.97715.jpg'

        # prepare image data
        data = dict(img=image_file)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]

        # [1, 19, 720, 1280], before softmax
        with torch.no_grad():
            image_feature = model.whole_inference(data['img'][0], data['img_metas'][0], True)

        points = np.fromfile(bin_file, dtype=np.float32).reshape([-1,4])  # n*4
        feature_info = np.zeros([points.shape[0], 19], dtype = np.float32)  # n*19
        points_painting = np.column_stack((points, feature_info))  # n*23
        points_T = np.transpose(points)   # 4*n        
        points_T[3, :] = 1.0  # for calculation

        # # lidar2camera
        points_T_camera = np.dot(ex_matrix, points_T)  # 3*n

        # camera2pixel
        pixel = np.dot(in_matrix, points_T_camera).T  # n*3
        pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]  # n*2
        pixel_xy = np.around(pixel_xy).astype(int)  # n*2

        for j in range(pixel_xy.shape[0]):
            if pixel_xy[j][0] >= 0 and pixel_xy[j][0] < 1280 \
                    and pixel_xy[j][1] >= 0 and pixel_xy[j][1] <720 \
                    and points_T_camera[2, j] > 0:
                for index in range(4, 22):
                    points_painting[j, index] = image_feature[0, index-4, pixel_xy[j][1], pixel_xy[j][0]]
        points_painting.tofile(bin_file_new)

def generate_painting_bin(work_dir):
    folders = []
    for root, dirs, files in os.walk(work_dir):
        if 'bin_v1' in dirs:
            folders.append(str(root))
    for folder in folders:
        if folder.split("/")[-2] == '3-29':
            time_start = time.time()    
            print('processing ', folder)
            bin_folder = os.path.join(folder, 'bin_v1')
            bin_painting_folder = os.path.join(folder, 'bin_painting_seg_feature')
            generate(folder, bin_folder, bin_painting_folder)
            time_end = time.time()
            print('time costs',time_end-time_start,'s')


work_dir = "/remote-home/share/SHTperson"
ex_matrix = np.array([[0.00852965, -0.999945, -0.00606215, 0.0609592],
                      [-0.0417155, 0.00570127, -0.999113, -0.144364],
                      [0.999093, 0.00877497, -0.0416646, -0.0731114]])
in_matrix = np.array([[683.8, 0.0, 673.5907],
                     [0.0, 684.147, 372.8048],
                     [0.0, 0.0, 1.0]])
config_file = '/root/PointPainting/painting/mmseg/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py'
checkpoint_file = '/root/PointPainting/painting/mmseg/checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
model = init_segmentor(config_file, checkpoint_file,
                            device='cuda:2')  # TODO edit here if you want to use different device
cfg = model.cfg
device = next(model.parameters()).device  # model device
# build the data pipeline
test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
test_pipeline = Compose(test_pipeline)
generate_painting_bin(work_dir)


