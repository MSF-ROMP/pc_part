import numpy as np
import cv2
import os


work_dir = "/remote-home/share/SHTperson"
ex_matrix = np.array([[0.00852965, -0.999945, -0.00606215, 0.0609592],
                      [-0.0417155, 0.00570127, -0.999113, -0.144364],
                      [0.999093, 0.00877497, -0.0416646, -0.0731114]])
in_matrix = np.array([[683.8, 0.0, 673.5907],
                     [0.0, 684.147, 372.8048],
                     [0.0, 0.0, 1.0]])
folders = []
for root, dirs, files in os.walk(work_dir):
    if 'bin_v1' in dirs:
        folders.append(str(root))

# ???
# '/remote-home/share/SHTperson/4-1/9' image problem

for folder in folders:
    # if folder.split("/")[-2] == '4-1':
    #     continue
    print("processing: ", folder)
    bin_folder = os.path.join(folder, 'bin_v1')
    image_folder = os.path.join(folder, 'left')
    file_list = os.listdir(bin_folder)
    image_file_list = os.listdir(image_folder)
    file_list.sort()
    image_file_list.sort()
    if file_list[0] == '.DS_Store':
        continue
    data_path = os.path.join(bin_folder, file_list[0])
    image_path = os.path.join(image_folder, image_file_list[0])

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


    # visualization 
    image = cv2.imread(image_path)
    # if image == None:
    #     print("No, ", folder)
    #     continue
    point_color = (0, 0, 255) # BGR

    for j in range(n):
        cv2.circle(image, (pixel_xy_image[j, 0], pixel_xy_image[j, 1]), 1, point_color, -1)
    
    path_list = bin_folder.split("/")
    # import pdb
    # pdb.set_trace()
    cv2.imwrite(os.path.join('/remote-home/share/SHTperson/vis/', 'scene_'+str(path_list[-2])+'_'+str(path_list[-3])+'.png'), image)
