import numpy as np

file_path = "/home/ashwin/Sensing_System_Project/pointnet/Semantic_Segmentation/pointnet/sem_seg/Area_6_office_18.npy"  # your file path
data = np.load(file_path, allow_pickle=True)

print("Data Shape before modification:", data.shape)
print("First few rows before modification:\n", data[:5])  # Showing first 5 rows for clarity

# Assuming the label is the last column, remove it
data = data[:, :-1]  # Keep all columns except the last one

print("Data Shape after modification:", data.shape)
print("First few rows after modification:\n", data[:5])

# Save the modified data back to a file (overwrite or save as a new file)
np.save(file_path, data)

