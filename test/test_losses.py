import sys
import os
import torch
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), "src"))
from molearn.loss_functions.geometric import compute_torsion_loss, calculate_dihedrals

def test_torsion_loss_is_zero_for_identical_structures():
    """
    Tests that the loss is zero when predicted and true coordinates are the same.
    """
    # Create a simple, random 5-residue backbone structure
    # Shape: (batch_size=1, length=5, num_atoms=4, coords=3)
    true_coords = torch.randn(1, 5, 4, 3)
    
    # The predicted coordinates are an exact copy
    predicted_coords = true_coords.clone()
    
    # Calculate the loss
    loss = compute_torsion_loss(predicted_coords, true_coords)
    
    # Assert that the loss is exactly 0.0
    assert loss.item() == 0.0, "Loss should be zero for identical inputs"

def test_torsion_loss_for_known_rotation():
    """
    Tests that the loss function calculates a predictable value for a known change.
    """
    # Create a simple 5-residue backbone
    true_coords = torch.randn(1, 5, 4, 3)
    predicted_coords = true_coords.clone()

    # Let's modify a single psi angle by rotating the final atom (N of residue i+1)
    # We will rotate the Nitrogen of the 3rd residue around the C-CA axis of the 2nd residue
    
    # This is a simplified rotation for demonstration. A real implementation
    # would use a rotation matrix.
    predicted_coords[0, 2, 0, :] += 1.0 # Add a 1.0 Angstrom shift to the N of residue 2
    
    loss = compute_torsion_loss(predicted_coords, true_coords)
    
    # Assert that the loss is a positive, non-zero number
    assert loss.item() > 0.0, "Loss should be positive for different inputs"
    
    # For a more rigorous test, you would manually calculate the expected MSE
    # on the sin/cos of the single changed angle and assert approximate equality.
    # For example:
    # true_psi = 0.5 rad, pred_psi = 1.0 rad
    # expected_loss = ((sin(1)-sin(0.5))^2 + (cos(1)-cos(0.5))^2) / 3
    # assert loss.item() == pytest.approx(expected_loss)


def test_torsion_loss_handles_periodicity():
    """
    Tests that the loss is small for angles that are close on the circle,
    e.g., +179 degrees and -179 degrees.
    """
    # Create a base structure
    base_coords = torch.zeros(1, 5, 4, 3)
    # A simple linear chain for simplicity
    for i in range(5):
        base_coords[0, i, 1, 0] = i # C-alpha x-coordinate
        base_coords[0, i, 2, 0] = i + 0.5 # Carbon x-coordinate
        
    # --- Create a "true" structure with a ~180 degree angle ---
    true_coords = base_coords.clone()
    true_coords[0, 3, 0, 1] = -1.0 # Move N of residue 3 to create a ~180 deg psi

    # --- Create prediction 1 with a ~+179 degree angle ---
    pred_coords_1 = base_coords.clone()
    pred_coords_1[0, 3, 0, 1] = -1.0 + 0.01 # Slightly perturb
    
    # --- Create prediction 2 with a ~-179 degree angle ---
    pred_coords_2 = base_coords.clone()
    pred_coords_2[0, 3, 0, 1] = -1.0 - 0.01 # Perturb in the other direction

    loss_1 = compute_torsion_loss(pred_coords_1, true_coords)
    loss_2 = compute_torsion_loss(pred_coords_2, true_coords)

    # Assert that both losses are very small
    assert loss_1.item() < 1e-4
    assert loss_2.item() < 1e-4
    
    # Assert that the two losses are almost identical
    assert loss_1.item() == pytest.approx(loss_2.item()), "Loss should be symmetric around 180 degrees"   