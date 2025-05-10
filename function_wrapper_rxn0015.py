import warnings
import logging
import time  # Import the time module
import torch

import numpy as np
from nequip.train import Trainer, Loss, Metrics
from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper
from ase import Atoms
from torch.autograd.functional import hessian


# Specific filter for the PyTorch version warning
warnings.filterwarnings("ignore", message=".*PyTorch version 2.6.0 found.*")
# General warning suppression
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

train_dir_path = 'nequip_data/rxn0015/rxn0015'
model, model_config = Trainer.load_model_from_training_session(
    traindir=train_dir_path, model_name="best_model.pth",
    # weights_only=False  # For PyTorch 2.6+ compatibility
)
model = model.to('cpu')
model.train()  # Set model to evaluation mode

full_data = np.load('nequip_data/benchmark_data/rxn0015-full.npz', allow_pickle=True)
test_idx = -1
positions = full_data['R'][test_idx]
flat_positions = positions.flatten()

atomic_numbers = full_data['z'].astype(int)
atoms = Atoms(
    positions=positions,
    numbers=atomic_numbers,
    cell=np.eye(3) * 20.0,  # Large box to avoid periodic boundary effects
    pbc=True  # Set periodic boundary conditions
)
type_names = model_config['type_names']
chemical_symbol_to_type = {sym: i for i, sym in enumerate(type_names)}
type_mapper = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)
r_max = model_config['r_max']
atomic_data = AtomicData.from_ase(atoms, r_max=r_max)
atomic_data = type_mapper(atomic_data)
input_dict = AtomicData.to_AtomicDataDict(atomic_data)
# input_dict[AtomicDataDict.POSITIONS_KEY] = input_dict[AtomicDataDict.POSITIONS_KEY].clone().detach().requires_grad_(True)


def get_model_rxn0015():
    return model

def nequip_potential_rxn0015(flat_positions):
    positions = flat_positions.reshape(-1, 3)
    # print(type(positions)) # Removed print statement for cleaner timing output
    input_dict[AtomicDataDict.POSITIONS_KEY] = positions #might need to require grad

    out = model(input_dict)


    return out[AtomicDataDict.TOTAL_ENERGY_KEY]#.item() #return the energy

def grad_potential_rxn0015(flat_positions):
    positions = flat_positions.reshape(-1, 3)
    input_dict[AtomicDataDict.POSITIONS_KEY] = positions #might need to require grad
    out = model(input_dict)
    return -out[AtomicDataDict.FORCE_KEY].flatten().to(dtype=torch.float32)



def extract_reaction_coordinates(position_vector):
    """
    Extract O-H bond length and C-O-H angle from a flattened position vector
    
    Parameters:
    position_vector: 1D numpy array of length 27 (9 atoms × 3 coordinates)
    
    Returns:
    oh_bond_length: Bond length between O(0) and H(6)
    coh_angle: Angle in degrees for C(1)-O(0)-H(6)
    """
    # Reshape the flat vector to (9, 3) for xyz coordinates
    positions = position_vector.reshape(-1, 3)
    
    # Extract atomic positions
    o_pos = positions[0]  # O atom at index 0
    c_pos = positions[1]  # C atom at index 1
    h_pos = positions[6]  # H atom at index 6
    
    # Calculate O-H bond length
    oh_bond_length = np.linalg.norm(o_pos - h_pos)
    
    # Calculate C-O-H angle
    # Vector from O to C
    oc_vector = c_pos - o_pos
    # Vector from O to H
    oh_vector = h_pos - o_pos
    
    # Normalize vectors
    oc_vector = oc_vector / np.linalg.norm(oc_vector)
    oh_vector = oh_vector / np.linalg.norm(oh_vector)
    
    # Calculate angle using dot product
    dot_product = np.dot(oc_vector, oh_vector)
    # Clip to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    
    # Convert to degrees
    coh_angle = np.degrees(angle_rad)
    
    return oh_bond_length, coh_angle



print(flat_positions.shape)
oh_bond_length, coh_angle = extract_reaction_coordinates(flat_positions)
print(oh_bond_length, coh_angle)



