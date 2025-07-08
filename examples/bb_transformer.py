import sys
import os

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.data import PDBData
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.models.transformer import Autoencoder
import torch

def main():
    # ==================================================================
    # 1. Experiment Configuration
    # ==================================================================

    exp_id = "Exp-001_Transformer"
    
    # Architectural hyperparameters
    model_params = {
        "L": 437, # Sequence length of MurD
        "node_dim": 32,
        "pair_dim": 32,
        "latent_dim": 2,
        "num_blocks": 2,
        "num_heads": 4,
        "dropout_p": 0.1
    }

    # Training hyperparameters
    training_params = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "validation_split": 0.1,
        "manual_seed": 25
    }

    # Regularization hyperparameters
    regularization_params = {
        "weight_decay": 1e-2
    }

    # Loss function weights (target ratios for dynamic scaling)
    loss_weights = {
        "w_rec": 1.0, 
        "w_dist": 0.5,
        "w_torsion": 0.2,
        "w_phys": 0.1
    }

    physics_params = {
        "start_physics_at": 10,
        "soft_NB": True
    }

    # ==================================================================
    # 2. Data Pipeline Setup
    # ==================================================================
    print(f"--- Setting up data for {exp_id} ---")

    # Instantiate PDBData with the transformer flag set to True
    data = PDBData(for_transformer=True)

    data.import_pdb(filename=["./data/MurD_closed_selection.pdb", "./data/MurD_open_selection.pdb"],
                    topology=None)
    data.fix_terminal()
    data.atomselect(atoms=["N", "CA", "C", "O"])

    # ==================================================================
    # 3. Trainer and Model Preparation
    # ==================================================================
    print("--- Preparing Trainer and Model ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = OpenMM_Physics_Trainer(device=device)

    # Pass data and training params to the trainer
    trainer.set_data(
        data, 
        batch_size=training_params["batch_size"], 
        validation_split=training_params["validation_split"],
        manual_seed=training_params["manual_seed"]
    )
    
    # Prepare the physics engine. This method is assumed to be updated
    # to handle the dummy std and initialize the OpenMM engine correctly.
    trainer.prepare_physics(**physics_params)

    # Set the new Autoencoder model with its specific hyperparameters
    trainer.set_autoencoder(Autoencoder, **model_params)
    
    # Prepare the optimizer, passing regularization params and loss weights
    trainer.prepare_optimiser(
        lr=training_params["learning_rate"],
        weight_decay=regularization_params["weight_decay"],
        loss_weights=loss_weights
    )

    # Print a summary of the network parameters
    print("--- Model Architecture Summary ---")
    network_summary = trainer.get_network_summary()
    for component, params in network_summary.items():
        print(f"{component:<25}: {params:,}")
    print("------------------------------------")

    # ==================================================================
    # 4. Training Loop with Organized Checkpointing
    # ==================================================================
    print(f"--- Starting Training Loop for {exp_id} ---")

    # Best practice: Create a unique folder for each experiment run
    output_folder = f"results/{exp_id}"
    os.makedirs(output_folder, exist_ok=True)
    
    runkwargs = dict(
        log_filename=os.path.join(output_folder, "training_log.dat"),
        checkpoint_folder=output_folder,
    )

    # The robust early stopping logic from the original script
    best = 1e24
    while True:
        # The trainer's run method will now handle the 4-component loss
        trainer.run(max_epochs=32 + trainer.epoch, **runkwargs)
        if not best > trainer.best:
            print("Early stopping criteria met.")
            break
        best = trainer.best
        
    print(f"\nTraining complete for {exp_id}.")
    print(f"Best validation loss: {trainer.best}")
    print(f"Best model saved to: {trainer.best_name}")


if __name__ == "__main__":
    main()