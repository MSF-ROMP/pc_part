import numpy as np
import pickle
import torch

pkl_path = "/remote-home/share/work_dir/11-16-fcos3d/result.pkl"
val_path = "/remote-home/share/SHTperson/save_info/sht_infos_val_2345678.pkl"

calib = np.array([[683.8, 0., 673.5907, 0.],
                [0., 684.147, 372.8048, 0.],
                [0., 0., 1., 0.]], np.float32)
ex_matrix = np.array([[0.00852965, -0.999945, -0.00606215, 0.0609592],
                    [-0.0417155, 0.00570127, -0.999113, -0.144364],
                    [0.999093, 0.00877497, -0.0416646, -0.0731114]], dtype=np.float32)

with open(pkl_path, "rb") as f:
    pred = pickle.load(f)

with open(val_path, "rb") as f:
    val = pickle.load(f)

print(len(pred), len(val))

preds_list = []
scores_list = []
annos_list = []
path_list = []
img_path_list = []
occ_list = []

val_num = len(val)
for i in range(val_num):
    pred_it = pred[i]["img_bbox"]
    scores = pred_it["scores_3d"].numpy()
    preds = pred_it["boxes_3d"].tensor.numpy()[:, :7]

    select_ind = scores > 0.2
    preds = preds[select_ind]
    scores = scores[select_ind]

    loc = preds[:, :3] - np.array([[0., 0.5, 0.]])*preds[:, [5, 3, 4]]
    R, T = ex_matrix[:, :3], ex_matrix[:, 3:]
    inv_RT = np.linalg.inv(R.T)
    loc = np.matmul((loc - T.T), inv_RT)
    preds[:, :3] = loc

    print(preds)
    print(pred_it.keys())
    exit(0)
    
    info = val[i]
    img_path = info["image"]["image_path"]
    pc_path = info["point_cloud"]["point_cloud_path"]

    anns = info["annos"]
    pos_gt = np.array(anns["position"])
    dim_gt = np.array(anns["dimensions"])
    rot_gt = np.array(anns["rotation"])
    bbox2d_gt = np.array(anns["image_bbox"]["2D"])
    occ_gt = np.array(anns["image_bbox"]["occlusion"])

    image_ids = np.array(anns["image_id"])
    item_ids = np.array(anns["tracking"]["item_id"])
    assert image_ids.shape[0] == bbox2d_gt.shape[0], "{} is not consistent to {}".format(image_ids.shape, bbox2d_gt.shape)
    assert item_ids.shape[0] == pos_gt.shape[0], "{} is not consistent to {}".format(item_ids.shape, pos_gt.shape)
    _, image_index, pc_index = np.intersect1d(image_ids, item_ids, assume_unique=False, return_indices=True)

    pos_gt = pos_gt[pc_index]
    dim_gt = dim_gt[pc_index]
    rot_gt = rot_gt[pc_index]
    occ_gt = occ_gt[image_index]
    annos = np.concatenate([pos_gt, dim_gt, rot_gt[..., np.newaxis]], axis=1)

    preds_list.append(preds.tolist())
    scores_list.append(scores.tolist())
    path_list.append(pc_path)
    annos_list.append(annos.tolist())
    occ_list.append(occ_gt.tolist())
    img_path_list.append(img_path)
    # print(annos)
    print(i)
    # exit(0)

output_dict = {"predict":preds_list, "score":scores_list, "path":path_list, "annos":annos_list, "occlusion":occ_list, "ima_path":img_path_list}

with open("/remote-home/share/SHTperson/result/result-mono3d_fcos3d_add.pkl", "wb") as f:
    pickle.dump(output_dict, f)