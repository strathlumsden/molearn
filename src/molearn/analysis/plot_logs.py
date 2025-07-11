import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys

def plot_loss_curves(log_file_path: str):
    """
    Reads a training log file and plots the training and validation loss curves.

    Args:
        log_file_path (str): The path to the training_log.dat file.
    """
    # --- 1. Load the Data ---
    # Use pandas to read the comma-separated log file into a DataFrame.
    try:
        df = pd.read_csv(log_file_path)
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file_path}'")
        sys.exit(1)

    # --- 2. Create the Plots ---
    # We create a 2x2 grid of subplots to visualize all key metrics.
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training and Validation Metrics for {os.path.basename(log_file_path)}', fontsize=16)

    # --- Plot 1: Total Validation Loss (The main metric for early stopping) ---
    # We plot the raw validation loss to see the convergence trend.
    axes[0, 0].plot(df['epoch'], df['valid_loss'], label='Total Validation Loss', color='orange', marker='.')
    axes[0, 0].set_title('Total Validation Loss (Log-Sum)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (Log-Sum Scale)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # --- Plot 2: Reconstruction Loss (MSE) ---
    axes[0, 1].plot(df['epoch'], df['train_rec_loss'], label='Train MSE', color='blue', alpha=0.7)
    axes[0, 1].plot(df['epoch'], df['valid_rec_loss'], label='Validation MSE', color='blue', marker='.')
    axes[0, 1].set_title('Reconstruction Loss (MSE)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE (Å²)')
    axes[0, 1].set_yscale('log') # Log scale is useful for reconstruction losses
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # --- Plot 3: Geometric Losses (Distance & Torsion) ---
    axes[1, 0].plot(df['epoch'], df['train_dist_loss'], label='Train Dist Loss', color='green', alpha=0.7)
    axes[1, 0].plot(df['epoch'], df['valid_dist_loss'], label='Validation Dist Loss', color='green', marker='.')
    axes[1, 0].plot(df['epoch'], df['train_torsion_loss'], label='Train Torsion Loss', color='red', alpha=0.7)
    axes[1, 0].plot(df['epoch'], df['valid_torsion_loss'], label='Validation Torsion Loss', color='red', marker='.')
    axes[1, 0].set_title('Geometric Component Losses')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # --- Plot 4: Physics Loss ---
    axes[1, 1].plot(df['epoch'], df['train_phys_loss'], label='Train Physics Loss', color='purple', alpha=0.7)
    axes[1, 1].plot(df['epoch'], df['valid_phys_loss'], label='Validation Physics Loss', color='purple', marker='.')
    axes[1, 1].set_title('Physics Loss (Energy)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Energy')
    # Use a symmetric log scale for energy, which can be negative
    axes[1, 1].set_yscale('symlog')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # --- 3. Save and Show the Plot ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure in the same directory as the log file.
    output_dir = os.path.dirname(log_file_path)
    plot_filename = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to: {plot_filename}")
    plt.show()

def main():
    """
    Main function to parse command-line arguments and run the plotting script.
    """
    # Use argparse to create a simple command-line interface
    parser = argparse.ArgumentParser(description="Plot training and validation loss curves from a log file.")
    parser.add_argument("log_file", type=str, help="Path to the training_log.dat file from your experiment.")
    args = parser.parse_args()
    
    plot_loss_curves(args.log_file)

if __name__ == "__main__":
    main()
