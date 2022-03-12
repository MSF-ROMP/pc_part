import os
import numpy as np
import fire
import open3d as o3d
def read_pcd(filepath):
    lidar = []
    print(filepath)
    with open(filepath,'r') as f:
        pcd = o3d.io.read_point_cloud(filepath)
        # print(np.asarray(pcd.points))
        # colors = np.asarray(pcd.colors) * 255
        ## points = np.asarray(pcd.points)
    return np.asarray(pcd.points)


def convert(pcdfolder, binfolder):
    current_path = os.getcwd()
    ori_path = os.path.join(current_path, pcdfolder)
    file_list = os.listdir(ori_path)
    des_path = os.path.join(current_path, binfolder)
    if os.path.exists(des_path):
        pass

    
    else:
        os.makedirs(des_path)
    for file in file_list: 
        (filename,extension) = os.path.splitext(file)
        velodyne_file = os.path.join(ori_path, filename) + '.pcd'
        pl = read_pcd(velodyne_file)
        print(pl)
        pl = pl.reshape(-1, 4).astype(np.float32)
        velodyne_file_new = os.path.join(des_path, filename) + '.bin'
        pl.tofile(velodyne_file_new)
    
if __name__ == "__main__":
    fire.Fire() 