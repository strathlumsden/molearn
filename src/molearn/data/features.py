import torch
import torch.nn.functional as F
from typing import Tuple

def calculate_dihedrals(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """
    Calculates the dihedral angles for a set of 4 points in a vectorized manner.
    
    Args:
        p0, p1, p2, p3: Tensors of coordinates of shape (B, K, 3), where K is the number
                        of dihedrals to calculate.

    Returns:
        A tensor of dihedral angles in radians of shape (B, K).
    """
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 to prevent numerical instability
    b1 = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True).clamp(min=1e-8)

    # Vector cross products to find the normals to the two planes
    v = torch.cross(b0, b1)
    w = torch.cross(b1, b2)

    x = torch.sum(v * w, dim=-1)
    y = torch.sum(torch.cross(v, w) * b1, dim=-1)

    return torch.atan2(y, x)


def get_backbone_torsion_features(backbone_coords: torch.Tensor) -> torch.Tensor:
    """
    Calculates and featurizes the phi, psi, and omega backbone torsion angles.

    Args:
        backbone_coords: A tensor of shape (B, L, 4, 3) with N, CA, C, O coordinates.

    Returns:
        A tensor of shape (B, L, 6) containing the [sin, cos] of phi, psi, and omega.
    """
    # Select atoms for torsion calculations, note the slicing to handle adjacent residues
    # Atoms for phi: C(i-1), N(i), CA(i), C(i)
    phi = calculate_dihedrals(
        p0=backbone_coords[:, :-1, 2],  # C from previous residue
        p1=backbone_coords[:, 1:, 0],   # N from current residue
        p2=backbone_coords[:, 1:, 1],   # CA from current residue
        p3=backbone_coords[:, 1:, 2],   # C from current residue
    )

    # Atoms for psi: N(i), CA(i), C(i), N(i+1)
    psi = calculate_dihedrals(
        p0=backbone_coords[:, :-1, 0],  # N from current residue
        p1=backbone_coords[:, :-1, 1],  # CA from current residue
        p2=backbone_coords[:, :-1, 2],  # C from current residue
        p3=backbone_coords[:, 1:, 0],   # N from next residue
    )

    # Atoms for omega: CA(i), C(i), N(i+1), CA(i+1)
    omega = calculate_dihedrals(
        p0=backbone_coords[:, :-1, 1],  # CA from current residue
        p1=backbone_coords[:, :-1, 2],  # C from current residue
        p2=backbone_coords[:, 1:, 0],   # N from next residue
        p3=backbone_coords[:, 1:, 1],   # CA from next residue
    )

    # Featurize angles using sin and cos
    phi_feat = torch.stack([torch.sin(phi), torch.cos(phi)], dim=-1)
    psi_feat = torch.stack([torch.sin(psi), torch.cos(psi)], dim=-1)
    omega_feat = torch.stack([torch.sin(omega), torch.cos(omega)], dim=-1)

    # Pad to restore original sequence length L.
    # Phi is missing at the first residue, Psi and Omega at the last.
    phi_padded = F.pad(phi_feat, (0, 0, 1, 0), "constant", 0)
    psi_padded = F.pad(psi_feat, (0, 0, 0, 1), "constant", 0)
    omega_padded = F.pad(omega_feat, (0, 0, 0, 1), "constant", 0)
    
    # Concatenate all features
    return torch.cat([phi_padded, psi_padded, omega_padded], dim=-1)


def get_distance_features(coords1: torch.Tensor, coords2: torch.Tensor, embedding_module: torch.nn.Module) -> torch.Tensor:
    """
    Calculates and embeds the pairwise distances between two sets of points.

    Args:
        coords1: A tensor of coordinates of shape (B, L, 3).
        coords2: A tensor of coordinates of shape (B, L, 3).
        embedding_module: A PyTorch module (e.g., Gaussian RBF layer) that embeds
                          scalar distances into feature vectors.

    Returns:
        A tensor of shape (B, L, L, num_features) representing the embedded distance map.
    """
    # Calculate pairwise Euclidean distances
    dmap = torch.cdist(coords1, coords2, p=2.0)
    
    # Pass distances through the embedding module
    dmap_embedded = embedding_module(dmap)
    
    return dmap_embedded