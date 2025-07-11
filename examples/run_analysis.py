import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from molearn.analysis.transformer_analyser import TransformerAnalysis
from molearn.models.transformer import Autoencoder
from molearn.data import PDBData

def main():
    """
    Main script to load a trained model and evaluate it on a test set.
    """
    # --- Configuration ---
    exp_id = "Exp-001_Baseline"
    checkpoint_path = f"results/{exp_id}/best_model.ckpt" # Example path
    

    # Define paths to all datasets needed for analysis
    open_state_path = "./data/MurD_open_selection.pdb"
    closed_state_path = "./data/MurD_closed_selection.pdb"
    transition_path = "./data/MurD_closed_apo_selection.pdb"
    
    # --- Load Model and Data ---
    print("--- Loading model and test set ---")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_params = checkpoint['network_kwargs']
    
    model = Autoencoder(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    analysis = TransformerAnalysis()
    analysis.set_network(model)

    # --- Load All Datasets for Analysis ---
    print("--- Loading and preparing datasets ---")
    
    # Load open states
    open_data = PDBData(for_transformer=True)
    open_data.import_pdb(open_state_path)
    open_data.fix_terminal()
    open_data.atomselect(atoms=["N", "CA", "C", "O"])
    open_data.prepare_dataset()
    analysis.set_dataset(key="open_state", data=open_data)

    # Load closed states
    closed_data = PDBData(for_transformer=True)
    closed_data.import_pdb(closed_state_path)
    closed_data.fix_terminal()
    closed_data.atomselect(atoms=["N", "CA", "C", "O"])
    closed_data.prepare_dataset()
    analysis.set_dataset(key="closed_state", data=closed_data)

    # Load transition path test set
    transition_data = PDBData(for_transformer=True)
    transition_data.import_pdb(filename=transition_path)
    transition_data.fix_terminal()
    transition_data.atomselect(atoms=["N", "CA", "C", "O"])
    transition_data.prepare_dataset()
    analysis.set_dataset(key="transition_path", data=transition_data)
    
    
    # --- Calculate and Report Metrics ---
    print("\n--- Calculating Evaluation Metrics ---")

    num_params = analysis.num_trainable_params()
    print(f"Trainable Parameters: {num_params:,}")
    print("------------------------------------")

    rmsd_values = analysis.get_error(key="transition_path")
    mean_rmsd = rmsd_values.mean()
    std_rmsd = rmsd_values.std()
    print(f"Reconstruction RMSD: {mean_rmsd:.3f} ± {std_rmsd:.3f} Å")

    dope_scores = analysis.get_dope(key="transition_path")
    mean_dope = dope_scores['decoded_dope'].mean()
    std_dope = dope_scores['decoded_dope'].std()
    print(f"Decoded DOPE Score: {mean_dope:.2f} ± {std_dope:.2f}")

    rama_scores = analysis.get_ramachandran(key="transition_path")
    favored_percent = (rama_scores['decoded_favored'].mean() / rama_scores['decoded_total'].mean()) * 100
    outlier_percent = (rama_scores['decoded_outliers'].mean() / rama_scores['decoded_total'].mean()) * 100
    print(f"Ramachandran - Favored: {favored_percent:.2f}% | Outliers: {outlier_percent:.2f}%")

    # --- Latent Space Visualization ---
    print("\n--- Generating Latent Space Visualizations ---")
    
    # Define the data to plot for our key analyses
    plot_data_config = [
        {'key': 'open_state', 'label': 'Open States', 'color': 'royalblue', 'plot_type': 'scatter', 's': 5},
        {'key': 'closed_state', 'label': 'Closed States', 'color': 'firebrick', 'plot_type': 'scatter', 's': 5},
        {'key': 'transition_path', 'label': 'Transition Path', 'color': 'limegreen', 'plot_type': 'path'}
    ]

    # Generate and save the PCA plot
    analysis.plot_latent_space(
        plot_data=plot_data_config,
        reduction='pca',
        fit_on_keys=['open_state', 'closed_state'], # Fit PCA on training data only
        output_filename=f"results/{exp_id}/latent_space_pca.png"
    )

    # Generate and save the t-SNE plot
    analysis.plot_latent_space(
        plot_data=plot_data_config,
        reduction='tsne',
        fit_on_keys=['open_state', 'closed_state'], # Fit t-SNE on training data only
        output_filename=f"results/{exp_id}/latent_space_tsne.png"
    )

    # --- Example of "Richness Check" Analysis ---
    # Color the closed states by simulation time (assuming it's sequential)
    closed_coords = analysis.get_dataset("closed_state")
    time_color_values = np.arange(len(closed_coords))
    
    analysis.plot_latent_space(
        plot_data=[
            {'key': 'closed_state', 'label': 'Closed States', 'plot_type': 'scatter', 's': 10}
        ],
        reduction='pca',
        fit_on_keys=['closed_state'],
        color_by_property={'key': 'closed_state', 'values': time_color_values, 'label': 'Simulation Time'},
        output_filename=f"results/{exp_id}/latent_space_richness_check.png"
    )


if __name__ == "__main__":
    main()
