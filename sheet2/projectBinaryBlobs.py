from matplotlib import pyplot as plt
import numpy as np
import os

# calculate the vicinity matrix of the binary files
name_to_follow_matrix = {}
RESOURCES = "sheet2/BinaryBlobs"

for filename in os.listdir(RESOURCES):
    with open(os.path.join(RESOURCES,filename),"rb") as file:
        data = file.read()
        matrix = np.zeros((256,256))
        previous_byte = 0b00000000
        for byte in data:
            matrix[previous_byte,byte] = matrix[previous_byte,byte] + 1
            previous_byte = byte
        name_to_follow_matrix[filename] = matrix

name_to_vicinity_matrix = {}
for key, value in name_to_follow_matrix.items():
    name_to_vicinity_matrix[key] = np.add(value,np.transpose(value))
    
# get max absolute eigenvalue
name_to_sorted_eig_value = {}
for key, value in name_to_vicinity_matrix.items():
    eigenvals = np.linalg.eigvals(value)
    sorted_abs_eigenvals =sorted(np.absolute(eigenvals))
    name_to_sorted_eig_value[key] = sorted_abs_eigenvals
    
# visualize in 3x3 grid
fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharey=True)

# Assuming 'name_to_sorted_eig_value' is a dictionary where each value is a list of points to be plotted
for ax, (key, value) in zip(axs.flatten(), name_to_sorted_eig_value.items()):
    x_values = range(len(value))  # Create a range for x values
    ax.scatter(x_values, value)  # Create a scatter plot
    ax.set_title(key)  # Set the title of the plot to the key

plt.tight_layout()  # Adjust the layout so that there's no overlap between the plots
plt.savefig("test2.png")  # Display the plots