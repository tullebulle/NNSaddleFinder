from potential_energy_function import PotentialEnergyFunction
from saddlefinder import NNSaddleFinder
import torch
from utils import run_saddle_search, plot_saddle_search_results
import matplotlib.pyplot as plt


pot_func= PotentialEnergyFunction.load('models/potential_energy_function_spline.pkl')


results = run_saddle_search(pot_func, torch.tensor([1.0, 90.0],), iterations=40000, step_size=0.05, momentum=0.4)
fig, ax = plot_saddle_search_results(pot_func, results)
plt.show()





