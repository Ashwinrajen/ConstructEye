import numpy as np
import argparse

# Argument parser to specify the .npy file path
parser = argparse.ArgumentParser(description="Load and print X, Y, Z from an .npy file")
parser.add_argument("--npy_file", required=True, help="Path to the .npy file")
args = parser.parse_args()

# Load the .npy file
data_label = np.load(args.npy_file)  # Assuming data is stored in (N, M) format

# Extract X, Y, Z coordinates (assuming they are in the first 3 columns)
x_data = data_label[:, 0]  # X values
y_data = data_label[:, 1]  # Y values
z_data = data_label[:, 2]  # Z values

# Print the extracted values
print("X Coordinates:", x_data)
print("Y Coordinates:", y_data)
print("Z Coordinates:", z_data)

