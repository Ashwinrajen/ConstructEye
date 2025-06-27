import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Load data from file
data = np.loadtxt('/home/ashwin/Sensing_System_Project/sensing_ws/reference_model_office.txt')

# Extract XYZ coordinates
xyz = data[:, 0:3]
pred_class = data[:, 7].astype(int)

# Generate colors based on class labels
num_classes = np.unique(pred_class).size
colormap = plt.get_cmap("tab20", num_classes)
colors = np.array([colormap(c % num_classes)[:3] for c in pred_class])

# Create Open3D Point Cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Create a visualization window with a larger bounding box
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Extended Point Cloud View", width=1200, height=800)

# Add the point cloud to the visualizer
vis.add_geometry(pcd)

# Get the view control object to adjust visualization parameters
view_control = vis.get_view_control()

# Set a larger bounding box for better visualization
pcd_bbox = pcd.get_axis_aligned_bounding_box()
pcd_bbox.scale(2.5, center=pcd_bbox.get_center())  # Scale up by 2.5x
vis.get_render_option().background_color = np.array([0, 0, 0])  # Set background to black

# Update visualization
vis.run()
vis.destroy_window()

