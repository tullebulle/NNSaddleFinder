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

train_dir_path = 'nequip_data/aspirin/minimal'
model, model_config = Trainer.load_model_from_training_session(
    traindir=train_dir_path, model_name="best_model.pth",
    # weights_only=False  # For PyTorch 2.6+ compatibility
)
model = model.to('cpu')
model.train()  # Set model to evaluation mode

test_data = np.load('nequip_data/benchmark_data/aspirin_ccsd-test.npz', allow_pickle=True)
test_idx = 0
positions = test_data['R'][test_idx]
flat_positions = positions.flatten()

atomic_numbers = test_data['z'].astype(int)
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


def get_model_asprin():
    return model

def nequip_potential_aspirin(flat_positions):
    positions = flat_positions.reshape(-1, 3)
    # print(type(positions)) # Removed print statement for cleaner timing output
    input_dict[AtomicDataDict.POSITIONS_KEY] = positions #might need to require grad

    out = model(input_dict)


    return out[AtomicDataDict.TOTAL_ENERGY_KEY]#.item() #return the energy

def grad_potential_aspirin(flat_positions):
    positions = flat_positions.reshape(-1, 3)
    input_dict[AtomicDataDict.POSITIONS_KEY] = positions #might need to require grad
    out = model(input_dict)
    return -out[AtomicDataDict.FORCE_KEY].flatten().to(dtype=torch.float32)




# print(flat_positions.shape)