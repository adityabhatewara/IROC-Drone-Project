'''
NASA's Mars drone Ingenuity is very similar to our drone Apollo. 
It has similar sensors and it needs an Algorithm to find safe landing spots too. 
I'm Trying to implement the same Algorithm they used on Apollo. 
'''

'''
About Algorithm: It basically finds the best fit plane in each kernel 
and then finds the error/ cost by the distance of each point in the kernel 
from the plane, similar to linear regression. Then then apply thresholds on this cost and also the slope of the plane.
'''


'''
There are a few issues with using this on Apollo. One is that the actual Algorithm 
implemented by them might be coded in cuda or c++ with heavy optimizations to reduce runtime.
This is a bit computationally heavy.
'''

import numpy as np
import open3d as o3d
import math
import subprocess
from scipy.ndimage import generic_filter
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cluster import DBSCAN

# command = [
#     "rtabmap-export",
#     "--cloud",
#     "--output", "cfi",
#     "--input", "/home/orin/.ros/rtabmap.db"
# ]

GRID_WIDTH = 200
GRID_HEIGHT = 200
SCALE_FACTOR = 10
GRID_OFFSET = 100
KERNEL_SIZE = 7

SLOPE_THRESHOLD = 10  # degrees
RESIDUAL_THRESHOLD = 0.05
Z_HEIGHT_THRESHOLD = 0.3

class CloudProcessor():
    def __init__(self):
        # self.save_map()
        self.z_array = np.full((GRID_WIDTH, GRID_HEIGHT), -100.0, dtype=float)
        self.all_points = np.array([[0, 0, 0], [0, 0, 0]])
        self.safe_spots = [0, 0, 0]

        self.pcd = o3d.io.read_point_cloud("/home/adityabhatewara/Desktop/Anveshak_stuff/iroc/util1.ply")
        self.safe_pcd = o3d.geometry.PointCloud()
        self.safe_np = np.array([[0, 0, 0]])
        self.points = np.asarray(self.pcd.points)
        self.skip_count = 0
        self.cloud_point_detection()

    # def save_map(self):
    #     subprocess.run(command, check=True)
    #     print("Point cloud exported successfully.")

    def cloud_point_detection(self):
        self.z_array.fill(-100.0)
        valid = np.all(np.isfinite(self.points), axis=1)
        points = self.points[valid]

        xi = np.round(points[:, 0] * SCALE_FACTOR).astype(int) + GRID_OFFSET
        yi = np.round(points[:, 1] * SCALE_FACTOR).astype(int) + GRID_OFFSET
        z = np.round(points[:, 2], 2)

        mask = (xi >= 0) & (xi < GRID_WIDTH) & (yi >= 0) & (yi < GRID_HEIGHT)
        xi, yi, z = xi[mask], yi[mask], z[mask]
        for x_val, y_val, z_val in zip(xi, yi, z):
            if self.z_array[x_val, y_val] == -100.0:
                self.z_array[x_val, y_val] = z_val

        print("cloud arrangement is over")
        self.safe_point_detection()

    def plane_fit_filter(self, z_block):
        size = int(np.sqrt(len(z_block)))
        if size % 2 == 0:
            return np.inf
        
        kernel_half = size // 2
        grid_x, grid_y = np.meshgrid(
            np.arange(-kernel_half, kernel_half + 1),
            np.arange(-kernel_half, kernel_half + 1),
            indexing='ij')

        x = grid_x.flatten()
        y = grid_y.flatten()
        z = z_block.flatten()
        
        valid = z != -100
        if np.max(z) >= Z_HEIGHT_THRESHOLD:
            return np.inf
        if np.count_nonzero(valid) < size:
            self.skip_count += 1
            return np.inf

        X = np.column_stack((x[valid], y[valid]))
        Z = z[valid]
        if np.var(Z) > 0.005:
            return np.inf
        else:
            return 0
        model = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor())
        model.fit(X, Z)

        a, b = model.named_steps['ransacregressor'].estimator_.coef_[1:]

        tilt_angle = np.rad2deg(np.arctan(np.sqrt(a**2 + b**2)))
        Z_pred = model.predict(X)
        residuals = Z - Z_pred
        rms = np.sqrt(np.mean(residuals**2))

        if tilt_angle > SLOPE_THRESHOLD or rms > RESIDUAL_THRESHOLD:
            return np.inf
        
        print(np.var(Z))
        return 0

    def safe_point_detection(self):
        self.safe_spots.clear()

        plane_map = generic_filter(
            self.z_array,
            lambda block: self.plane_fit_filter(block),
            size=KERNEL_SIZE,
            mode='constant',
            cval=-100
        )

        for i in range(KERNEL_SIZE//2, GRID_WIDTH - KERNEL_SIZE//2):
            for j in range(KERNEL_SIZE//2, GRID_HEIGHT - KERNEL_SIZE//2):
                zi = self.z_array[i, j]
                if zi == -100:
                    continue
                xi, yi = ((i - GRID_OFFSET) / SCALE_FACTOR, (j - GRID_OFFSET) / SCALE_FACTOR)

                if plane_map[i, j] == np.inf or abs(zi) > Z_HEIGHT_THRESHOLD:
                    continue

                self.safe_spots.append([xi, yi, zi])

        print(f"Detected {len(self.safe_spots)} safe spots")

        self.safe_np = np.array(self.safe_spots) if len(self.safe_spots) > 0 else np.array([[0, 0, 0]])
        self.safe_np[:,2] += 0.2
        self.safe_pcd.points = o3d.utility.Vector3dVector(self.safe_np)
        self.safe_pcd.paint_uniform_color([0, 1, 0])

        self.final_safe = self.find_safe_group_centroids(self.safe_np)
        self.final_safe_np = np.array(self.final_safe)

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        geometry = [self.pcd, self.safe_pcd, coordinate_frame]

        for c in self.final_safe:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.paint_uniform_color([1.0, 1.0, 0.0])
            sphere.translate(c)
            geometry.append(sphere)

        o3d.visualization.draw_geometries(geometry)
        return self.final_safe

    def find_safe_group_centroids(self, safe_spot_list, eps=0.5, min_samples=50):
        if len(safe_spot_list) == 0:
            return []

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(safe_spot_list[:, :2])
        labels = clustering.labels_

        centroids = []
        for label in np.unique(labels):
            if label == -1:
                continue

            cluster_points = safe_spot_list[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
            print(f"Cluster {label}: {len(cluster_points)} points → Centroid: {centroid}")
            print(self.skip_count)
        return centroids

def main(args=None):
    cloud_processor = CloudProcessor()

if __name__ == '__main__':
    main()
