import numpy as np
import open3d as o3d

# Load the point cloud data (Assumed format: X, Y, Z, R, G, B)
file_path = "/home/ashwin/Sensing_System_Project/sensing_ws/reference_model_office.npy"  # Change this
point_cloud = np.load(file_path)

# Extract XYZ and normalize RGB (0-255 → 0-1)
xyz = point_cloud[:, :3]
rgb = point_cloud[:, 3:6] / 255.0

# Create Open3D Point Cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

# Visualize
o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")


