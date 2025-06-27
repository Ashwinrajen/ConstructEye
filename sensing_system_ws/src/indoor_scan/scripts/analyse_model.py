#!/usr/bin/env python3

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from indoor_scan.srv import ProcessPointCloud, ProcessPointCloudResponse
import os
import rospkg
import subprocess
import open3d as o3d
import matplotlib.pyplot as plt
from indoor_scan.msg import PointCloudDetections
from std_msgs.msg import Header


CLASS_NAMES = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
]

CLASS_COLORS = {
    "ceiling": [0.5, 0.5, 0.5],  # Gray
    "floor": [0.0, 0.0, 1.0],  # Blue
    "wall": [1.0, 0.0, 0.0],  # Red
    "beam": [1.0, 1.0, 0.0],  # Yellow
    "column": [1.0, 0.65, 0.0],  # Orange
    "window": [0.0, 1.0, 1.0],  # Cyan
    "door": [0.5, 0.0, 0.0],  # Dark Red
    "table": [0.6, 0.3, 0.0],  # Brown
    "chair": [0.0, 1.0, 0.0],  # Green
    "sofa": [0.5, 0.0, 0.5],  # Purple
    "bookcase": [0.3, 0.2, 0.0],  # Dark Brown
    "board": [0.0, 0.5, 0.0],  # Dark Green
    "clutter": [1.0, 0.0, 1.0],  # Magenta
}

COLOR_NAMES = {
    "ceiling": "Gray",
    "floor": "Blue",
    "wall": "Red",
    "beam": "Yellow",
    "column": "Orange",
    "window": "Cyan",
    "door": "Dark Red",
    "table": "Brown",
    "chair": "Green",
    "sofa": "Purple",
    "bookcase": "Dark Brown",
    "board": "Dark Green",
    "clutter": "Magenta",
}


class_colors = {}  # Store assigned colors for each class


class PointCloudSaver:
    """ROS Node for Capturing, Saving, Processing, and Visualizing Point Clouds.

    This class implements a ROS service that listens for a trigger request, captures a
    PointCloud2 message from a subscribed topic, saves it as a .npy file, runs inference
    on the saved file, and visualizes the output. The service runs continuously and waits
    for the next call after the visualization window is closed.
    """

    def __init__(self):
        """Initializes the ROS node and sets up the service.

        - Creates a ROS service `/save_pointcloud` that listens for trigger requests.
        - Initializes subscriber for the PointCloud2 topic but does not start it immediately.
        - Logs the readiness of the PointCloud saver service.
        """
        rospy.init_node("pointcloud_saver", anonymous=True)

        # Create a service that listens for a trigger request
        self.service = rospy.Service("/save_pointcloud", ProcessPointCloud, self.service_callback)
        self.detection_publisher = rospy.Publisher("/pointcloud_detections", PointCloudDetections, queue_size=10)

        # Subscriber for the PointCloud2 topic
        self.pc_sub = None
        self.latest_pc = None
        self.reference_class_names_list = []
        self.prediction_class_names_list = []
        self.reference_class_counts_list = []
        self.prediction_class_counts_list = []
        self.reference_color_names_list = []
        self.prediction_color_names_list = []
        self.reference_file_name = ""
        self.prediction_file_name = ""

        rospy.loginfo("PointCloud Saver Service is Ready...")

    def analyze_predictions(self, pred_file, is_reference=False):
        """Reads the prediction file, prints detected objects, and publishes results as a ROS message."""
        data = np.loadtxt(pred_file)
        pred_labels = data[:, 7].astype(int)

        unique_classes, counts = np.unique(pred_labels, return_counts=True)

        class_colors = {}  # Store assigned colors for visualization
        color_rgb_list = []  # Store RGB for visualization (Open3D)

        print("\nDetected Objects in Scene:")
        for cls_id, count in zip(unique_classes, counts):
            class_name = str(CLASS_NAMES[cls_id]) if cls_id < len(CLASS_NAMES) else f"Unknown ({cls_id})"
            color_name = str(COLOR_NAMES.get(class_name, "White"))  # Convert to Python string for ROS message
            rgb_color = CLASS_COLORS.get(class_name, [1.0, 1.0, 1.0])  # Keep RGB values for Open3D

            print(f"- {class_name}: {count} points | Color: {color_name}")

            # Store for ROS message (string-based)
            if is_reference:
                self.reference_class_names_list.append(class_name)
                self.reference_class_counts_list.append(int(count))
                self.reference_color_names_list.append(color_name)
            else:
                self.prediction_class_names_list.append(class_name)
                self.prediction_class_counts_list.append(int(count))
                self.prediction_color_names_list.append(color_name)

            # Store for Open3D visualization (float-based)
            class_colors[cls_id] = rgb_color  # Ensure Open3D only gets numerical values
            color_rgb_list.append(rgb_color)  # Keep RGB for visualization

        return class_colors  # Only return numerical RGB values for Open3D

    def process_reference_pointcloud(self, reference_npy_path):
        """Runs inference on the provided reference point cloud and visualizes it."""

        # Get the ROS package path dynamically
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("indoor_scan")  # Get path of the ROS package

        # Save reference predictions in a separate directory
        dump_dir = os.path.join(package_path, "miscellaneous/reference_predictions")
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        # Generate an incremental filename for reference point cloud predictions
        base_name = "reference_pointCloud"
        file_index = 1
        while os.path.exists(os.path.join(dump_dir, f"{base_name}_{file_index}.txt")):
            file_index += 1

        reference_pred_filename = os.path.join(dump_dir, f"{base_name}_{file_index}.txt")

        # Run the inference script for reference point cloud
        command = [
            "python3",
            os.path.join(package_path, "scripts/DNN_Inference", "inference_kd_search.py"),
            "--model_path",
            os.path.join(package_path, "miscellaneous/models", "model.ckpt"),
            "--dump_dir",
            dump_dir,
            "--input_npy_file",
            reference_npy_path,
            "--output_pred_file",
            reference_pred_filename,
        ]

        try:
            rospy.loginfo("Running inference on Reference PointCloud...")
            subprocess.run(command, check=True)

            # Analyze reference predictions
            self.analyze_predictions(reference_pred_filename, is_reference=True)

            rospy.loginfo(f"Reference PointCloud processed and saved at: {reference_pred_filename}")

            # Visualize Reference PointCloud
            self.visualize_prediction(reference_pred_filename, file_index, is_reference=True)

        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Inference script failed for reference point cloud: {e}")

    def service_callback(self, request):
        """Handles service requests to process a point cloud file or capture a live point cloud."""
        rospy.loginfo("Service Called: Checking for file path...")

        if request.file_path:  # If a file path is provided, process it as a reference point cloud
            reference_npy_path = request.file_path
            if os.path.exists(reference_npy_path):
                rospy.loginfo(f"Processing Reference PointCloud from: {reference_npy_path}")

                # Process and visualize reference point cloud
                self.process_reference_pointcloud(reference_npy_path)

                rospy.loginfo("Reference PointCloud visualization closed. Now capturing RTAB scan...")

                # Proceed to capture live scan after closing the visualization window
                if self.pc_sub is None:
                    self.pc_sub = rospy.Subscriber("/rtabmap/scan_map", PointCloud2, self.pointcloud_callback)

                rospy.sleep(1)  # Give some time to receive a message

                if self.latest_pc is not None:
                    self.save_pointcloud(self.latest_pc)
                    return ProcessPointCloudResponse(success=True, message="Live PointCloud saved and processed.")
                else:
                    return ProcessPointCloudResponse(success=False, message="No PointCloud received.")

            else:
                rospy.logerr(f"Reference file not found: {reference_npy_path}")
                return ProcessPointCloudResponse(success=False, message="Reference file not found.")

        # If no file path, proceed with capturing live point cloud directly
        if self.pc_sub is None:
            self.pc_sub = rospy.Subscriber("/rtabmap/scan_map", PointCloud2, self.pointcloud_callback)

        rospy.sleep(1)  # Give some time to receive a message

        if self.latest_pc is not None:
            self.save_pointcloud(self.latest_pc)
            print("Clearing List....")
            return ProcessPointCloudResponse(success=True, message="Live PointCloud saved and processed.")
        else:
            return ProcessPointCloudResponse(success=False, message="No PointCloud received.")

    def pointcloud_callback(self, msg):
        """Callback function that stores the latest received PointCloud2 message.

        - Gets triggered whenever a new PointCloud2 message is published.
        - Stores the message for processing when the service is called.
        """
        rospy.loginfo("PointCloud received!")
        self.latest_pc = msg

    def visualize_prediction(self, pred_file, file_index, is_reference):
        """Visualizes the point cloud with assigned colors for each object class."""

        # Load the prediction data
        data = np.loadtxt(pred_file)

        # Extract XYZ coordinates and class labels
        xyz = data[:, 0:3]
        pred_class = data[:, 7].astype(int)

        # Get the fixed colors assigned during analysis
        if is_reference:
            self.reference_class_names_list = []
            self.reference_color_names_list = []
            self.reference_class_counts_list = []
            class_colors = self.analyze_predictions(pred_file, is_reference=True)  # Get assigned colors
            self.reference_file_name = str(pred_file)
        else:
            self.prediction_class_names_list = []
            self.prediction_class_counts_list = []
            self.prediction_color_names_list = []
            class_colors = self.analyze_predictions(pred_file, is_reference=False)  # Get assigned colors
            self.prediction_file_name = str(pred_file)

        # Assign colors based on the class label
        colors = np.array([class_colors.get(cls, [1.0, 1.0, 1.0]) for cls in pred_class])  # Default: White if missing

        # Create Open3D Point Cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud Visualization", width=1200, height=800)

        # Add the point cloud to the visualizer
        vis.add_geometry(pcd)

        # Set visualization parameters
        vis.get_render_option().background_color = np.array([0, 0, 0])  # Set background to black

        # Run visualization and block execution until closed
        vis.run()
        vis.destroy_window()

        rospy.loginfo("Visualization closed. Proceeding with next step.")

        # Create and populate the ROS message
        msg = PointCloudDetections()
        msg.header.stamp = rospy.Time.now()
        msg.reference_file = self.reference_file_name
        msg.reference_counts = self.reference_class_counts_list
        msg.reference_colors = self.reference_color_names_list
        msg.reference_class_names = self.reference_class_names_list
        msg.prediction_file = self.prediction_file_name
        msg.prediction_counts = self.prediction_class_counts_list
        msg.prediction_colors = self.prediction_color_names_list
        msg.prediction_class_names = self.prediction_class_names_list

        self.detection_publisher.publish(msg)  # Publish detection message

    def save_pointcloud(self, pc_msg):
        """Processes the captured point cloud, saves it, runs inference, and visualizes results.

        1. **Converts PointCloud2 to a NumPy array (XYZ + RGB placeholder).**
        2. **Generates a unique filename** (`input_pointCloud_X.npy`) to avoid overwriting previous data.
        3. **Saves the point cloud** in the package’s `miscellaneous/inputs` directory.
        4. **Runs an inference script (`inference_kd_search.py`)** with the saved file.
        5. **Loads the prediction output (`pred_pointCloud_X.txt`)** and visualizes it using Open3D.
        6. **Waits for the user to close the visualization before allowing new service calls.**
        """
        pc_data = []
        for p in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            r, g, b = 0, 0, 0  # Always set RGB to 0
            pc_data.append([x, y, z, r, g, b])

        pc_array = np.array(pc_data, dtype=np.float32)

        # Get the ROS package path dynamically
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("indoor_scan")  # Get path of the ROS package

        # Save the .npy file
        # Generate an incremental filename for input point cloud
        base_name = "input_pointCloud"
        file_index = 1

        while os.path.exists(os.path.join(package_path, "miscellaneous/inputs", f"{base_name}_{file_index}.npy")):
            file_index += 1

        input_filename = os.path.join(package_path, "miscellaneous/inputs", f"{base_name}_{file_index}.npy")
        np.save(input_filename, pc_array)
        rospy.loginfo(f"PointCloud saved to: {input_filename}")

        # Construct paths relative to the package
        model_path = os.path.join(package_path, "miscellaneous/models", "model.ckpt")
        dump_dir = os.path.join(package_path, "miscellaneous/predictions")

        # Run the inference script automatically
        command = [
            "python3",
            os.path.join(package_path, "scripts/DNN_Inference", "inference_kd_search.py"),
            "--model_path",
            model_path,
            "--dump_dir",
            dump_dir,
            "--input_npy_file",
            input_filename,
            "--output_pred_file",
            os.path.join(dump_dir, f"pred_pointCloud_{file_index}.txt"),
        ]

        try:
            rospy.loginfo("Running inference on saved PointCloud...")
            subprocess.run(command, check=True)

            pred_file = os.path.join(dump_dir, f"pred_pointCloud_{file_index}.txt")

            # Print detected objects and colors
            self.analyze_predictions(pred_file, is_reference=False)

            rospy.loginfo("Inference completed successfully. Visualizing output...")
            self.visualize_prediction(pred_file, file_index, is_reference=False)

        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Inference script failed: {e}")


if __name__ == "__main__":
    """Starts the ROS node and keeps it running.

    - Initializes the `PointCloudSaver` instance.
    - Calls `rospy.spin()` to keep the node running indefinitely.
    - Handles `ROSInterruptException` to allow graceful shutdown.
    """
    try:
        saver = PointCloudSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
