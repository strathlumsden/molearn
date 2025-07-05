import torch
import torch.nn.functional as F

def calc_dmap(xyz: torch.Tensor) -> torch.Tensor:
    """ Helper function to compute pairwise distance matrices. """
    return torch.cdist(xyz, xyz, p=2.0)

def compute_distance_loss(predicted_coords: torch.Tensor, 
                          true_coords: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Squared Error between the C-alpha distance matrices
    of predicted and true coordinates.

    Args:
        predicted_coords (torch.Tensor): Shape (B, L, 4, 3)
        true_coords (torch.Tensor): Shape (B, L, 4, 3)

    Returns:
        A scalar tensor representing the distance matrix loss.
    """
    # Select C-alpha coordinates (index 1)
    pred_ca = predicted_coords[:, :, 1, :]
    true_ca = true_coords[:, :, 1, :]
    
    # Calculate the distance matrices
    dmap_pred = calc_dmap(pred_ca)
    dmap_true = calc_dmap(true_ca)
    
    # Compute the Mean Squared Error between the two matrices
    loss = F.mse_loss(dmap_pred, dmap_true)
    
    return loss


def calculate_dihedrals(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """
    Calculates the dihedral angles for a set of 4 points in a vectorized manner.
    (This is the same helper function as defined previously).
    """
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True).clamp(min=1e-8)

    v = torch.linalg.cross(b0, b1)
    w = torch.linalg.cross(b1, b2)

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.linalg.cross(v, w) * b1, dim=-1)

    return torch.atan2(y, x)

def compute_torsion_loss(predicted_coords: torch.Tensor, 
                         true_coords: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Squared Error between the sin/cos representations of
    the backbone torsion angles for predicted and true coordinates.

    Args:
        predicted_coords (torch.Tensor): Shape (B, L, 4, 3)
        true_coords (torch.Tensor): Shape (B, L, 4, 3)

    Returns:
        A scalar tensor representing the torsion angle reconstruction loss.
    """
    # --- 1. Calculate Predicted Torsions ---
    pred_phi = calculate_dihedrals(predicted_coords[:, :-1, 2], predicted_coords[:, 1:, 0], predicted_coords[:, 1:, 1], predicted_coords[:, 1:, 2])
    pred_psi = calculate_dihedrals(predicted_coords[:, :-1, 0], predicted_coords[:, :-1, 1], predicted_coords[:, :-1, 2], predicted_coords[:, 1:, 0])
    pred_omega = calculate_dihedrals(predicted_coords[:, :-1, 1], predicted_coords[:, :-1, 2], predicted_coords[:, 1:, 0], predicted_coords[:, 1:, 1])

    # --- 2. Calculate True Torsions ---
    true_phi = calculate_dihedrals(true_coords[:, :-1, 2], true_coords[:, 1:, 0], true_coords[:, 1:, 1], true_coords[:, 1:, 2])
    true_psi = calculate_dihedrals(true_coords[:, :-1, 0], true_coords[:, :-1, 1], true_coords[:, :-1, 2], true_coords[:, 1:, 0])
    true_omega = calculate_dihedrals(true_coords[:, :-1, 1], true_coords[:, :-1, 2], true_coords[:, 1:, 0], true_coords[:, 1:, 1])

    # --- 3. Featurize all angles as [sin, cos] vectors ---
    pred_torsions = torch.stack([
        torch.sin(pred_phi), torch.cos(pred_phi),
        torch.sin(pred_psi), torch.cos(pred_psi),
        torch.sin(pred_omega), torch.cos(pred_omega)
    ], dim=-1) # Shape: (B, L-2, 6) or (B, L-1, 6)

    true_torsions = torch.stack([
        torch.sin(true_phi), torch.cos(true_phi),
        torch.sin(true_psi), torch.cos(true_psi),
        torch.sin(true_omega), torch.cos(true_omega)
    ], dim=-1)

    # --- 4. Create a mask to handle undefined terminal angles ---
    # Phi is undefined for the first residue, Psi/Omega for the last.
    # The intersection of these is the central L-2 residues.
    # We can simplify by just comparing the L-2 valid phi angles.
    
    # A more robust mask for all angles:
    # We compare the sin/cos of phi, psi, omega.
    # Phi is defined for residues 1 to L-1. Psi/Omega for 0 to L-2.
    # To compare all, we can compute loss on each and average.
    
    # Let's compute loss for each angle type on its valid range
    loss_phi = F.mse_loss(torch.sin(pred_phi), torch.sin(true_phi)) + F.mse_loss(torch.cos(pred_phi), torch.cos(true_phi))
    loss_psi = F.mse_loss(torch.sin(pred_psi), torch.sin(true_psi)) + F.mse_loss(torch.cos(pred_psi), torch.cos(true_psi))
    loss_omega = F.mse_loss(torch.sin(pred_omega), torch.sin(true_omega)) + F.mse_loss(torch.cos(pred_omega), torch.cos(true_omega))

    # --- 5. Return the average loss ---
    return (loss_phi + loss_psi + loss_omega) / 3.0

