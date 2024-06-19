import os
from safetensors.torch import load_file, save_file
import torch

# Get all safetensors files in the current directory
safetensors_files = [f for f in os.listdir('.') if f.endswith('.safetensors')]

# Process each safetensors file
for file in safetensors_files:
    # Load the file
    tensors = load_file(file)
    
    # Halve the weights and convert to FP16
    for key in tensors.keys():
        tensors[key] = (tensors[key] / 2).half()
    
    # Save as a new file, adding 'half_' prefix to the filename
    new_file = f'half_{file}'
    save_file(tensors, new_file)
    
    print(f'{file} has been processed and saved as {new_file}')

