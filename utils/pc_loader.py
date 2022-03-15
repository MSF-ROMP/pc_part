import numpy as np
import open3d
import json

Color = {
    'red': [255, 0, 0],
    'blue': [0, 0, 255],
    'green': [0, 255, 0],
    'yellow': [255, 255, 0],
    'gold': [255, 204, 0],
    'black': [0, 0, 0],
    'gray': [128, 128, 128],
    'white': [255, 255, 255]
}


class Visualize:
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    def add_points(self, points: np.array, color='blue'):
        color_list = np.array([Color[color] for _ in range(points.shape[0])], dtype=float)
        color_list = color_list / 255
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.colors = open3d.utility.Vector3dVector(color_list)
        # np_points = np.asarray(pcd.points)
        self.vis.add_geometry(pcd)

    def add_lines(self, points: np.array, lines: np.array, color='red'):
        color_list = np.array([Color[color] for _ in range(lines.shape[0])], dtype=float)
        color_list = color_list / 255
        line_pcd = open3d.geometry.LineSet()
        line_pcd.lines = open3d.utility.Vector2iVector(lines)
        line_pcd.points = open3d.utility.Vector3dVector(points)
        line_pcd.colors = open3d.utility.Vector3dVector(color_list)
        self.vis.add_geometry(line_pcd)


    def run(self):
        # self.vis.poll_events()
        # self.vis.update_renderer()
        self.vis.run()


with open("../data/3-22 场景8-1.json") as f:
    info = json.load(f)

person = []
for item in info["frames"][0]["items"]:
    person.append((np.array([item["position"]["x"], item["position"]["y"], item["position"]["z"]]),
                   np.array([item["boundingbox"]["x"], item["boundingbox"]["y"], item["boundingbox"]["z"]])))


data_path = "../data/3-22/1/bin_v1/1616406608.796536064.bin"
points = np.fromfile(data_path, dtype=np.float32).reshape([-1,4])
pc = []
human = []

# for position, bbox in person:
#     print(position, bbox)


def in_box(position, bbox, point):
    lower_bound = position - bbox / 2
    upper_bound = position + bbox / 2
    return lower_bound[0] < point[0] < upper_bound[0] and lower_bound[1] < point[1] < upper_bound[1] and lower_bound[2] < point[2] < upper_bound[2]


def compute_bbox(position, bbox):
    points = np.array([[position[0] - bbox[0] / 2, position[1] - bbox[1] / 2, position[2] - bbox[2] / 2],
                       [position[0] + bbox[0] / 2, position[1] - bbox[1] / 2, position[2] - bbox[2] / 2],
                       [position[0] - bbox[0] / 2, position[1] + bbox[1] / 2, position[2] - bbox[2] / 2],
                       [position[0] - bbox[0] / 2, position[1] - bbox[1] / 2, position[2] + bbox[2] / 2],
                       [position[0] + bbox[0] / 2, position[1] + bbox[1] / 2, position[2] + bbox[2] / 2],
                       [position[0] - bbox[0] / 2, position[1] + bbox[1] / 2, position[2] + bbox[2] / 2],
                       [position[0] + bbox[0] / 2, position[1] - bbox[1] / 2, position[2] + bbox[2] / 2],
                       [position[0] + bbox[0] / 2, position[1] + bbox[1] / 2, position[2] - bbox[2] / 2]])
    lines = np.array([[0, 1], [0, 2], [1, 7], [2, 7],
                      [1, 6], [4, 7], [2, 5], [0, 3],
                      [3, 6], [3, 5], [4, 6], [4, 5]])
    return points, lines


for point in points:
    h = False
    for position, bbox in person:
        if in_box(position, bbox, point):
            human.append(point[:3])
            h = True
    if not h:
        pc.append(point[:3])

mesh = open3d.io.read_triangle_mesh("../data/1_1616406608.97715_0.obj")
mesh.compute_vertex_normals()
open3d.visualization.draw_geometries([mesh])


# vis = Visualize()
# for position, bbox in person:
#     bb_points, bb_lines = compute_bbox(position, bbox)
#     vis.add_lines(bb_points, bb_lines, 'green')
# # vis.add_points(np.array(pc), 'blue')
# vis.add_points(np.array(human), 'red')
# vis.run()
