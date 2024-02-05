import pathlib
from typing import List

import torchvision
import torch

import pkg_motion_prediction.pre_load as pre_load
from pkg_motion_prediction.utils import utils_np
from pkg_motion_prediction.data_handle import data_handler

################################################## Defining motion ########################################
## Data given
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor

# Assuming your data is in the format [(timestamp1, x1, y1), (timestamp2, x2, y2), ...]
motion_data = [
    (1, 2, 3),
    (2, 5, 4),
    (3, 5, 6),
    (4, 5, 7),  # New data point
    (5, 7, 8),  # New data point
    (6, 8, 8),  # New data point
    (7, 9, 9), # New data point
    (8, 9, 10),# New data point
    (9, 10, 11),# New data point
    (10, 12, 12) # New data point
    # Add more data points as needed
]

# Separate the timestamps, x coordinates, and y coordinates into separate lists
timestamps, x_coords, y_coords = zip(*motion_data)

# Plot the x and y coordinates as a physical path
print("Plotting now...")
plt.figure(figsize=(10, 5))
plt.plot(x_coords, y_coords, marker='o')

# Annotate each point with its corresponding timestamp
for t, x, y in motion_data:
    plt.annotate(t, (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Physical Path of Motion')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.axis('equal')  # This ensures that one unit in x is the same as one unit in y
plt.show()
##########################################################################################

# Flatten your motion_data
flattened_data = [item for sublist in motion_data for item in sublist]

required_size = 3200
padded_data = flattened_data + [0] * (required_size - len(flattened_data))  # Padding with zeros

# Convert to tensor and reshape if necessary
input_tensor = torch.tensor(padded_data).float()
input_tensor = input_tensor.view(-1, 3200)  # Reshaping to match the model's expected input shape
print(input_tensor.shape)

# Now, input_tensor should be ready to be fed into your network
########################################################################################################

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]

class MmpInterface:
    def __init__(self, config_file_name: str):
        self._prt_name = 'MMPInterface'
        self.config = pre_load.load_config(ROOT_DIR, config_file_name)
        self.network_manager = pre_load.load_net(ROOT_DIR, config_file_name)

    def get_motion_prediction(self, input_traj: List[tuple], ref_image: torch.Tensor, pred_offset: int, rescale:float=1.0, batch_size:int=1) -> List[np.ndarray]:
        """Return a list of predicted positions.
        
        Arguments:
            input_traj: A list of tuples, each tuple is a coordinate (x, y).
            ref_image: A tensor of shape (height, width).
            pred_offset: The maximum number of steps to predict.
            rescale: The scale of the input trajectory.
            batch_size: The batch size for inference.

        Returns:
            A list (length is horizon) of predicted positions, each element is a ndarry with K (#hypo) rows.
        """
        if input_traj is None:
            return None
            
        input_traj = [[x*rescale for x in y] for y in input_traj]
        transform = torchvision.transforms.Compose([data_handler.ToTensor()])
        if not isinstance(ref_image, torch.Tensor):
            raise TypeError(f'The reference image should be a tensor, got {type(ref_image)}.')
        input_ = pre_load.traj_to_input(input_traj, ref_image=ref_image, transform=transform, obsv_len=self.config.obsv_len)
        
        hypos_list:List[np.ndarray] = []

        ### Batch inference
        input_all = input_.unsqueeze(0)
        for offset in range(1, pred_offset+1):
            input_[-1,:,:] = offset*torch.ones_like(input_[-1,:,:])
            input_all = torch.cat((input_all, input_.unsqueeze(0)), dim=0)
        input_all = input_all[1:]
        hyposM = torch.Tensor()
        for i in range(pred_offset//batch_size):
            input_batch = input_all[batch_size*i:batch_size*(i+1), :]
            try:
                hyposM = torch.concat((hyposM, self.network_manager.inference(input_batch)), dim=0)
            except:
                hyposM = self.network_manager.inference(input_batch)
        if pred_offset%batch_size > 0:
            input_batch = input_all[batch_size*(pred_offset//batch_size):, :]
            hyposM = torch.concat((hyposM, self.network_manager.inference(input_batch)), dim=0)
        for i in range(pred_offset):
            hypos_list.append(utils_np.get_closest_edge_point(hyposM[i,:].numpy(), 255 - ref_image.numpy()) / rescale)

        ### Single inference
        # for offset in range(1, pred_offset+1):
        #     input_[-1,:,:] = offset*torch.ones_like(input_[-1,:,:])

        #     hyposM = self.net.inference(input_.unsqueeze(0))[0,:]

        #     hyposM = utils_np.get_closest_edge_point(hyposM, 255 - ref_image) # post-processing
        #     hypos_list.append(hyposM/rescale)
        return hypos_list

# Assuming you have a configuration file name (replace 'your_config_file_name.yaml' with your actual config file name)
config_file_name = 'wsd_1t20_test.yaml'

# Initialize the MmpInterface class
mmp_interface = MmpInterface(config_file_name)

# Prepare the motion_data by extracting only the x and y coordinates
input_traj = [(x, y) for _, x, y in motion_data]

# Prepare a reference image as a torch.Tensor (replace this with your actual reference image)
# For demonstration, creating a dummy image of shape (224, 224)
ref_image = torch.rand((224, 224))

# Define the prediction offset, rescale factor, and batch size
pred_offset = 5  # For example, predict the next 5 positions
rescale = 1.0  # Assuming no rescaling is needed
batch_size = 1  # Assuming a batch size of 1

# Get the list of predicted positions
predicted_positions = mmp_interface.get_motion_prediction(input_traj, ref_image, pred_offset, rescale, batch_size)

# Print or process the predicted positions
print(predicted_positions)