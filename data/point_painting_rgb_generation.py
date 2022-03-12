import numpy as np
import os
from PIL import Image
import time


work_dir = "/remote-home/share/SHTperson"
ex_matrix = np.array([[0.00852965, -0.999945, -0.00606215, 0.0609592],
                      [-0.0417155, 0.00570127, -0.999113, -0.144364],
                      [0.999093, 0.00877497, -0.0416646, -0.0731114]])
in_matrix = np.array([[683.8, 0.0, 673.5907],
                     [0.0, 684.147, 372.8048],
                     [0.0, 0.0, 1.0]])
'''
data_path = '/remote-home/share/SHTperson/3-22/1/bin_v1/1616406608.796536064.bin'
points = np.fromfile(data_path, dtype=np.float32).reshape([-1,4])
points_T = np.transpose(points)
points_T[3, :] = 1.0


# lidar2camera
points_T_camera = np.dot(ex_matrix, points_T)

# camera2pixel
pixel = np.dot(in_matrix, points_T_camera).T
pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
pixel_xy = np.around(pixel_xy).astype(int)

n = 0
points_image = []
pixel_xy_image = []
for i in range(pixel_xy.shape[0]):
    if pixel_xy[i][0] >= 0 and pixel_xy[i][0] <= 1280 \
        and pixel_xy[i][1] >= 0 and pixel_xy[i][1] <=720 and points_T_camera[2, i] > 0:
        n += 1
        points_image.append(points_T_camera[0:3, i])
        pixel_xy_image.append(pixel_xy[i, :])
        # print(i, " ", pixel_xy[i])
points_image = np.array(points_image)
pixel_xy_image = np.array(pixel_xy_image)
print("number of pointclouds in image: ", n,"/", pixel.shape[0])
'''

'''
visualization 1
'''
# import numpy as np
# import cv2

# image_path = "/remote-home/share/SHTperson/3-22/1/left/1_1616406608.97715.jpg"
# image = cv2.imread(image_path)
# point_color = (0, 0, 255) # BGR

# for j in range(n):
#     cv2.circle(image, (pixel_xy_image[j, 0], pixel_xy_image[j, 1]), 1, point_color, -1)

# cv2.imwrite('/remote-home/share/SHTperson/vis2.png', image)


'''
visualization 2
'''
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt


# np.save("/remote-home/share/SHTperson/vis.npy", points_image)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = points_image[:, 0]
# y = points_image[:, 1]
# z = points_image[:, 2]

# ax.scatter(x, y, z, c='r', marker='.')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
# plt.savefig('/remote-home/share/SHTperson/vis.png')

'''
generate binary format point cloud with rgb channel
'''
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
            continue
        image_file = os.path.join(image_path, image_file_list[i])
        image = np.array(Image.open(image_file))

        # single frame test
        # bin_file = '/remote-home/share/SHTperson/3-22/1/bin_v1/1616406608.796536064.bin'
        # bin_file_new = '/remote-home/share/SHTperson/3-22/1/bin_painting/1616406608.796536064.bin'
        # image_file = '/remote-home/share/SHTperson/3-22/1/left/1_1616406608.97715.jpg'

        # # ???
        # if filename == '.DS_Store':
        #     continue

        points = np.fromfile(bin_file, dtype=np.float32).reshape([-1,4])  # n*4
        rgb_info = np.zeros([points.shape[0],3], dtype = np.float32)  # n*3
        points_painting = np.column_stack((points, rgb_info))  # n*7
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
                points_painting[j, 4] = image[pixel_xy[j][1], pixel_xy[j][0]][0]
                points_painting[j, 5] = image[pixel_xy[j][1], pixel_xy[j][0]][1]
                points_painting[j, 6] = image[pixel_xy[j][1], pixel_xy[j][0]][2]
                # print(j, points_painting[j, :])
        # import pdb
        # pdb.set_trace()
        # print(points_painting[12400, :])
        points_painting.tofile(bin_file_new)

def generate_painting_bin(work_dir):
    folders = []
    for root, dirs, files in os.walk(work_dir):
        if 'bin_v1' in dirs:
            folders.append(str(root))
    for folder in folders:
        # if folder.split("/")[-2] == '4-1':
        #     continue
        time_start=time.time()
        bin_folder = os.path.join(folder, 'bin_v1')
        bin_painting_folder = os.path.join(folder, 'bin_painting')
        generate(folder, bin_folder, bin_painting_folder)
        time_end=time.time()
        print('processing ', folder, 'time cost',time_end-time_start,'s')

generate_painting_bin(work_dir)