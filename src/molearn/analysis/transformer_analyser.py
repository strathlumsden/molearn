from __future__ import annotations
import os
from copy import deepcopy
import numpy as np
import torch.optim
from pathlib import Path
from typing import Union

try:
    # from modeller import *
    from modeller.selection import Selection
    from modeller.environ import Environ
    from modeller.scripts import complete_pdb
except Exception as e:
    print("Error importing modeller: ")
    print(e)

try:
    from ..scoring import Parallel_DOPE_Score
except ImportError as e:
    print(
        "Import Error captured while trying to import Parallel_DOPE_Score, it is likely that you dont have Modeller installed"
    )
    print(e)
try:
    from ..scoring import Parallel_Ramachandran_Score
except ImportError as e:
    print(
        "Import Error captured while trying to import Parallel_Ramachandran_Score, it is likely that you dont have cctbx/iotbx installed"
    )
    print(e)
from ..data import PDBData

from ..utils import as_numpy
from tqdm import tqdm
import warnings

from openmm.app.modeller import Modeller
from openmm.app.forcefield import ForceField
from openmm.app.pdbfile import PDBFile

# from openmm.app import PME
from openmm.app import NoCutoff
from openmm.openmm import VerletIntegrator
from openmm.app.simulation import Simulation
from openmm.unit import picoseconds

warnings.filterwarnings("ignore")

from molearn.analysis import MolearnAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class TransformerAnalysis(MolearnAnalysis):
    """
    A specialized subclass for analyzing the new transformer-based autoencoder.
    It inherits from MolearnAnalysis and overrides methods to handle the new
    data format (B,L,4,3) and model architecture.
    """
    def __init__(self):
        # Call the parent constructor
        super().__init__()
        # Set a flag to indicate the model type, if needed internally
        self.is_transformer_model = True

     # ## UPDATE: Added num_trainable_params method ##
    def num_trainable_params(self):
        """Returns the number of trainable parameters in the network."""
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    # ## UPDATE: Added get_encoded method ##
    def get_encoded(self, key):
        """Encodes a dataset using the transformer model."""
        if key not in self._encoded:
            assert key in self._datasets, f"Dataset key '{key}' not found."
            dataset = self.get_dataset(key)
            encoded = torch.empty(dataset.shape[0], self.network.encoder.output_projection.out_features)
            
            with torch.no_grad():
                 for i in tqdm(range(0, dataset.shape[0], self.batch_size), desc=f"Encoding {key}"):
                    batch_coords = dataset[i : i + self.batch_size].to(self.device)
                    # Use the model's dedicated .encode() method
                    encoded_batch = self.network.encode(batch_coords).cpu()
                    encoded[i : i + self.batch_size] = encoded_batch
            
            self._encoded[key] = encoded
        return self._encoded[key]

    def set_dataset(self, key: str, data: PDBData):
        """
        OVERRIDDEN METHOD: Sets data formatted for the transformer model.
        """
        assert data.for_transformer, "This method requires data processed with for_transformer=True"
        self._datasets[key] = data.dataset.float()
        
        # Store necessary metadata directly from the data object
        if not hasattr(self, "mol"): self.mol = data.frame()
        if not hasattr(self, "atoms"): self.atoms = data.atoms
        if not hasattr(self, "shape"): self.shape = (data.dataset.shape[1], data.dataset.shape[2], data.dataset.shape[3])

    def get_decoded(self, key):
        """
        OVERRIDDEN METHOD: Handles the transformer decoder's requirement for a
        reference structure to generate the 2D pair bias.
        """
        if key not in self._decoded:
            with torch.no_grad():
                encoded = self.get_encoded(key)
                reference_coords = self.get_dataset(key)
                decoded = torch.empty(encoded.shape[0], *self.shape).float()

                for i in tqdm(range(0, encoded.shape[0], self.batch_size), desc="Decoding Structures"):
                    batch_z = encoded[i : i + self.batch_size].to(self.device)
                    batch_ref = reference_coords[i : i + self.batch_size].to(self.device)
                    
                    decoded_batch = self.network.decode(batch_z, batch_ref).cpu()
                    decoded[i : i + self.batch_size] = decoded_batch
                
                self._decoded[key] = decoded
        return self._decoded[key]

    def get_error(self, key, align=True):
        """
        OVERRIDDEN METHOD: Correctly handles the (B,L,4,3) data format.
        """
        dataset = self.get_dataset(key)
        decoded = self.get_decoded(key)

        # Data is not normalized, so we can use it directly
        crd_ref = as_numpy(dataset)
        crd_mdl = as_numpy(decoded)
        
        err = []
        for i in range(dataset.shape[0]):
            # ## UPDATE: Create a fresh molecule object inside the loop ##
            # This prevents errors from persistent internal state.
            m = deepcopy(self.mol)

            # Reshape from (L,4,3) to (L*4, 3) for RMSD calculation
            ref = crd_ref[i].reshape(-1, 3)
            mdl = crd_mdl[i].reshape(-1, 3)

            m.coordinates = np.expand_dims(ref, 0)
            m.add_xyz(mdl)
            rmsd = m.rmsd(0, 1)
            err.append(rmsd)

        return np.array(err)

    def get_all_dope_score(self, tensor, refine=True, **kwargs):
        """
        OVERRIDDEN METHOD: Correctly handles reshaping transformer data for scoring.
        """
        if not hasattr(self, "dope_score_class"):
            self.dope_score_class = Parallel_DOPE_Score(self.mol, self.processes)

        results = []
        for frame in tqdm(tensor, desc="Calculating DOPE Scores"):
            # Reshape from (L,4,3) to (L*4, 3)
            coords_np = frame.reshape(-1, 3).cpu().numpy()
            results.append(self.dope_score_class.get_score(coords_np, refine=refine, **kwargs))
        
        return np.array([r.get() for r in results])

    def _ramachandran_score(self, frame):
        """
        ## NEW OVERRIDDEN METHOD ##
        This version correctly handles un-normalized data by passing it
        directly to the scoring function without multiplying by stdval.
        """
        if not hasattr(self, "ramachandran_score_class"):
            self.ramachandran_score_class = Parallel_Ramachandran_Score(
                self.mol, self.processes
            )
        
        # Ensure frame is a numpy array
        if isinstance(frame, torch.Tensor):
            f = frame.data.cpu().numpy()
        else:
            f = frame
        
        # Pass the coordinates directly without un-scaling
        return self.ramachandran_score_class.get_score(f)
    
    def get_all_ramachandran_score(self, tensor):
        """
        OVERRIDDEN METHOD: Correctly handles reshaping transformer data for Ramachandran scoring.
        """
        if not hasattr(self, "ramachandran_score_class"):
            self.ramachandran_score_class = Parallel_Ramachandran_Score(self.mol, self.processes)

        rama = dict(favored=[], allowed=[], outliers=[], total=[])
        results = []
        for frame in tqdm(tensor, desc="Calculating Ramachandran Scores"):
            # Reshape from (L,4,3) to (L*4, 3)
            coords_np = frame.reshape(-1, 3).cpu().numpy()
            results.append(self._ramachandran_score(coords_np))

        for r in results:
            favored, allowed, outliers, total = r.get()
            rama["favored"].append(favored)
            rama["allowed"].append(allowed)
            rama["outliers"].append(outliers)
            rama["total"].append(total)

        return {key: np.array(value) for key, value in rama.items()}

    def plot_latent_space(self, 
                          plot_data: list, 
                          reduction: str = None, 
                          fit_on_keys: list = None,
                          color_by_property: dict = None,
                          output_filename: str = 'latent_space.png'):
        """
        Generates a 2D plot of the latent space, with options for dimensionality
        reduction and coloring by physical properties.

        Args:
            plot_data (list): A list of dictionaries, each specifying a dataset to plot.
                              Example: [{'key': 'open', 'label': 'Open', 'color': 'blue', 'plot_type': 'scatter'}]
            reduction (str, optional): Method for dimensionality reduction ('pca' or 'tsne').
                                       Required if latent_dim > 2. Defaults to None.
            fit_on_keys (list, optional): List of dataset keys to use for fitting the
                                          reduction model (e.g., training data).
            color_by_property (dict, optional): Dictionary to color points by a property.
                                                Example: {'key': 'closed', 'values': time_array, 'label': 'Time'}
            output_filename (str): Path to save the output plot.
        """
        print(f"Generating plot for {output_filename}...")
        
        # --- 1. Prepare Latent Vectors and Dimensionality Reduction ---
        latent_vectors = {key: as_numpy(self.get_encoded(key)) for key in self._datasets.keys()}
        
        if reduction:
            if reduction.lower() == 'pca':
                if fit_on_keys is None:
                    raise ValueError("`fit_on_keys` must be provided when using PCA.")
                fit_data = np.concatenate([latent_vectors[key] for key in fit_on_keys])
                reducer = PCA(n_components=2)
                print(f"Fitting PCA on {fit_on_keys}...")
                reducer.fit(fit_data)
                projected_vectors = {key: reducer.transform(vec) for key, vec in latent_vectors.items()}
            
            elif reduction.lower() == 'tsne':
                # ## UPDATE: Corrected logic for t-SNE ##
                # Concatenate all datasets to be plotted into a single array
                all_keys = [config['key'] for config in plot_data]
                combined_data = np.concatenate([latent_vectors[key] for key in all_keys])
                
                reducer = TSNE(n_components=2, perplexity=30)
                print("Fitting and transforming with t-SNE...")
                projected_combined = reducer.fit_transform(combined_data)
                
                # Slice the projected data back into the original groups
                projected_vectors = {}
                current_idx = 0
                for key in all_keys:
                    num_points = len(latent_vectors[key])
                    projected_vectors[key] = projected_combined[current_idx : current_idx + num_points]
                    current_idx += num_points
            else:
                raise ValueError(f"Unknown reduction method: {reduction}")
        else:
            if list(latent_vectors.values())[0].shape[1] != 2:
                raise ValueError("Must specify a reduction method for latent spaces with dim > 2.")
            projected_vectors = latent_vectors


        # --- 2. Create Plot ---
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # --- 3. Plot Data from Configuration ---
        for config in plot_data:
            key = config['key']
            data_2d = projected_vectors[key]
            
            if config['plot_type'] == 'scatter':
                # Check if we need to color by a specific property
                if color_by_property and color_by_property['key'] == key:
                    c = color_by_property['values']
                    cmap = 'viridis'
                    label = None # Legend will be handled by colorbar
                else:
                    c = config.get('color', 'gray')
                    cmap = None
                    label = config.get('label')

                scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=c, cmap=cmap, label=label, s=config.get('s', 10), alpha=0.7)
                
            elif config['plot_type'] == 'path':
                ax.plot(data_2d[:, 0], data_2d[:, 1], color=config.get('color', 'green'), label=config.get('label'), lw=2)
        
        # --- 4. Add Colorbar if Needed ---
        if color_by_property:
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(color_by_property.get('label', 'Property Value'), fontsize=12)

        # --- 5. Formatting and Saving ---
        ax.set_title('Latent Space Visualization', fontsize=16, fontweight='bold')
        ax.set_xlabel('Latent Dimension 1' if reduction is None else f'{reduction.upper()} Component 1')
        ax.set_ylabel('Latent Dimension 2' if reduction is None else f'{reduction.upper()} Component 2')
        ax.legend()