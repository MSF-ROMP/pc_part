import os
import shutil

work_dir = "/remote-home/share/SHTperson"

def delete_bin_painting(work_dir):
    folders = []
    for root, dirs, files in os.walk(work_dir):
        if 'bin_painting_seg_prob' in dirs:
            folders.append(str(root))
    # print(folders)
    for folder in folders:
        bin_painting_folder = os.path.join(folder, 'bin_painting_seg_prob')
        shutil.rmtree(bin_painting_folder)
        # print(bin_painting_folder)
delete_bin_painting(work_dir)


