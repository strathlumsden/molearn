import torch
import torch.nn as nn

class GaussianSmearing(nn.Module):
    """
    Embeds scalar distances into a vector representation using Gaussian basis functions.
    
    This technique is a standard for representing distances in geometric deep learning
    models for molecular systems.
    """
    def __init__(self, start: float, stop: float, num_gaussians: int):
        """
        Args:
            start (float): The minimum distance to be considered (e.g., 0.0 Å).
            stop (float): The maximum distance (cutoff). Distances beyond this are not
                           meaningfully represented.
            num_gaussians (int): The number of Gaussian functions, which determines the
                                 output feature dimension.
        """
        super().__init__()
        
        # Pre-compute the centers of the Gaussian functions (mu).
        # These are linearly spaced from the start to the stop distance.
        centers = torch.linspace(start, stop, num_gaussians)
        
        # The width of the Gaussians (sigma) is fixed and determined by the spacing
        # between centers. This ensures good coverage of the entire distance range.
        self.sigma = (stop - start) / num_gaussians
        
        # Register centers as a non-trainable buffer. This ensures it's part of the
        # module's state and moves to the correct device (e.g., GPU) with the model.
        self.register_buffer('centers', centers)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Applies the Gaussian smearing to a tensor of distances.
        
        Args:
            distances (torch.Tensor): A tensor of distances, e.g., of shape (B, L, L).

        Returns:
            A tensor of embedded distances of shape (B, L, L, num_gaussians).
        """
        # Add a new dimension for broadcasting against the Gaussian centers
        distances = distances.unsqueeze(-1)
        
        # Calculate the value under each Gaussian function
        # The formula is exp(- (distance - center)^2 / (2 * sigma^2))
        exponent = -1 * (distances - self.centers)**2 / (2 * self.sigma**2)
        
        return torch.exp(exponent)
    

class AF2_PositionalEmbedding(nn.Module):
    """
    Implements the relative positional embedding from the AlphaFold2 paper.
    """
    def __init__(self, pair_embed_dim: int, pos_embed_r: int = 32):
        """
        Args:
            pair_embed_dim (int): The dimension of the final pair/edge representation.
            pos_embed_r (int): The maximum relative position to consider.
        """
        super().__init__()
        self.pos_embed_r = pos_embed_r
        # Create a learnable embedding for relative positions from -r to +r.
        # Total size is 2*r + 1 for positions, plus one for "too far".
        self.embedding = nn.Embedding(2 * pos_embed_r + 2, pair_embed_dim)

    def forward(self, residx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residx (torch.Tensor): A tensor of residue indices of shape (B, L).

        Returns:
            A tensor of positional embeddings of shape (B, L, L, pair_embed_dim).
        """
        B, L = residx.shape
        # Create a matrix of relative distances: d[i, j] = i - j
        d = residx[:, :, None] - residx[:, None, :]
        
        # Clip the distances to be within the range [-r, r]
        d_clipped = torch.clamp(d, -self.pos_embed_r, self.pos_embed_r)
        
        # Shift the indices to be non-negative for the embedding lookup
        # The "+ r + 1" index is used for all distances > r or < -r
        d_final = d_clipped + self.pos_embed_r
        
        # Use a mask for distances outside the range
        too_far = torch.abs(d) > self.pos_embed_r
        d_final[too_far] = 2 * self.pos_embed_r + 1
        
        # Look up the embeddings
        pos_embed = self.embedding(d_final)
        return pos_embed
    

