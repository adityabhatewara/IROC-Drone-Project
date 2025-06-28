import numpy as np
import open3d as o3d
import math
import subprocess
from scipy.ndimage import generic_filter
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

SLOPE_THRESHOLD = 0.25   # More conservative slope filter
Z_HEIGHT_THRESHOLD = 0.3   # Reject high/low points
VARIANCE_THRESHOLD = 0.007

class CloudProcessor():
    def __init__(self):
        # self.save_map()
        self.z_dict = {}
        self.z_array = np.full((GRID_WIDTH, GRID_HEIGHT), -100.0, dtype=float)
        self.all_points = np.array([[0, 0, 0], [0, 0, 0]])
        self.safe_spots = [0, 0, 0]

        self.pcd = o3d.io.read_point_cloud("/home/adityabhatewara/Desktop/Anveshak_stuff/iroc/util_better_cloud.ply")
        self.safe_pcd = o3d.geometry.PointCloud()
        self.safe_np = np.array([[0, 0, 0]])
        self.points = np.asarray(self.pcd.points)

        self.cloud_point_detection()

    # def save_map(self):
    #     subprocess.run(command, check=True)
    #     print("Point cloud exported successfully.")

    def cloud_point_detection(self):
        self.z_dict.clear()
        self.z_array.fill(-100.0)

        valid = np.all(np.isfinite(self.points), axis=1)
        points = self.points[valid]

        xi = np.round(points[:, 0] * SCALE_FACTOR).astype(int) + GRID_OFFSET
        yi = np.round(points[:, 1] * SCALE_FACTOR).astype(int) + GRID_OFFSET
        z = np.round(points[:, 2], 2)

        mask = (xi >= 0) & (xi < GRID_WIDTH) & (yi >= 0) & (yi < GRID_HEIGHT)
        xi, yi, z = xi[mask], yi[mask], z[mask]
        for x_val,y_val,z_val in zip(xi,yi,z):
            if self.z_array[x_val,y_val] == -100.0:
           
             self.z_array[x_val,y_val] = z_val
       
     
        print("cloud arrangement is over")
        self.safe_point_detection()

    def max_slope_filter(self, z_block):
        size = int(np.sqrt(len(z_block)))
        center_idx = len(z_block) // 2
        center = z_block[center_idx]
       
       
        z_min = np.min(z_block)
        z_max = np.max(z_block)
        
        if center == -100:
            return np.inf  # Center cell is invalid, cannot assess slope

        kernel_half = size // 2
        dx, dy = np.meshgrid(
            np.arange(-kernel_half, kernel_half + 1),
            np.arange(-kernel_half, kernel_half + 1),
            indexing='ij'
        )
        distances = np.sqrt(dx**2 + dy**2).flatten()
        distances[distances == 0] = 1e-6

        # Mask out invalid cells
        valid_mask = z_block != -100
        valid_mask[center_idx] = False  # Ignore center when computing slope to neighbors

        if not np.any(valid_mask):
            return np.inf  # No valid surrounding cells

        slopes = np.abs(z_block[valid_mask] - center) / distances[valid_mask]

        return np.max(slopes)  # Or np.max if you want to be stricter

    def height_variation(self, z_block):
        valid = z_block[z_block != -100]
        if len(valid) == 0:
            return np.inf
        return np.max(valid) - np.min(valid)
    

    def height_variance(self, z_block):
        valid = z_block[z_block != -100]  # Ignore empty grid cells
        if len(valid) == 0:
            return np.inf  # No good data
        return np.var(valid)  # Variance of heights

    def safe_point_detection(self):
        self.safe_spots.clear()

        slope_map= generic_filter(
            self.z_array,
            self.max_slope_filter,
            size=KERNEL_SIZE,
            mode='constant',
            cval=-100
        )
        dz_map = generic_filter(self.z_array, self.height_variation, size=KERNEL_SIZE, mode='constant', cval=-100)

        var_map = generic_filter(
            self.z_array,
            self.height_variance,
            size=KERNEL_SIZE,
            mode='constant',
            cval=-100
        )
        for i in range(KERNEL_SIZE//2, GRID_WIDTH - KERNEL_SIZE//2):
            for j in range(KERNEL_SIZE//2, GRID_HEIGHT - KERNEL_SIZE//2):
                zi = self.z_array[i, j]
                xi, yi = ((i - GRID_OFFSET) / SCALE_FACTOR, (j - GRID_OFFSET) / SCALE_FACTOR)

                if slope_map[i, j] > SLOPE_THRESHOLD or dz_map[i,j] > 0.3:    ### uncomment for using the slope method 
                    continue

                #if var_map[i, j] > VARIANCE_THRESHOLD:                         ### uncomment for using the variance method
                 #   continue

                self.safe_spots.append([xi, yi, zi])
        
        print(f"Detected {len(self.safe_spots)} safe spots")

        self.safe_np = np.array(self.safe_spots) if len(self.safe_spots) > 0 else np.array([[0, 0, 0]])
        #self.safe_np[:,2] += 0.5
       
        self.safe_pcd.points = o3d.utility.Vector3dVector(self.safe_np)
        self.safe_pcd.paint_uniform_color([0, 1, 0])
        #self.final_safe = self.safe_np
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

        return centroids


def main(args=None):
    cloud_processor = CloudProcessor()

if __name__ == '__main__':
    main()
