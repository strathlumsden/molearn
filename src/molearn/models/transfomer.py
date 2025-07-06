import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), "src"))
from molearn.data.features import calculate_dihedrals, get_backbone_torsion_features, get_distance_features
from molearn.models.embedding import GaussianSmearing, AF2_PositionalEmbedding


class MultiHeadAttention(nn.Module):
    """
    A multi-head attention layer that accepts a 2D pair bias.
    """
    def __init__(self, node_dim: int, pair_dim: int, num_heads: int, dropout_p: float = 0.1):
        super().__init__()
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        # A single linear layer to create Q, K, V projections efficiently
        self.qkv_linear = nn.Linear(node_dim, 3 * node_dim, bias=False)
        
        # A linear layer to project the 2D pair features into an attention bias
        self.bias_linear = nn.Linear(pair_dim, num_heads)
        
        # The final output projection layer
        self.out_linear = nn.Linear(node_dim, node_dim)

        # A dropout layer to regularize the attention weights
        self.attn_dropout = nn.Dropout(p=dropout_p)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Input s (nodes) shape: (B, L, node_dim)
        # Input z (pairs) shape: (B, L, L, pair_dim)
        B, L, _ = s.shape

        # 1. Project to Q, K, V and split into heads
        q, k, v = self.qkv_linear(s).chunk(3, dim=-1)
        # Reshape for multi-head processing
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape of q, k, v is now (B, num_heads, L, head_dim)

        # 2. Project 2D pair features to get attention bias
        bias = self.bias_linear(z).permute(0, 3, 1, 2)
        # bias shape is now (B, num_heads, L, L), matching the attention matrix

        # 3. Calculate biased attention scores
        # Scaling by sqrt(head_dim) is crucial for stable training
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores + bias
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # 4. Apply attention to V vectors
        s_update = torch.matmul(attn_weights, v)
        
        # 5. Combine heads and project to final output
        s_update = s_update.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_linear(s_update)

class FeedForward(nn.Module):
    """ A simple two-layer MLP with a GELU activation, used in transformer blocks. """
    def __init__(self, dim: int, multiplier: int = 4, dropout_p: float = 0.1):
        """
        Args:
            dim (int): The input and output dimension of the network.
            multiplier (int): The expansion factor for the hidden layer.
        """
        super().__init__()
        hidden_dim = dim * multiplier
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p)
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (B, L, dim)
        # Output shape: (B, L, dim)
        return self.net(x)
    

class TransformerBlock(nn.Module):
    """ A standard transformer block with pre-normalization. """
    def __init__(self, node_dim: int, pair_dim: int, num_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(node_dim)
        self.attn = MultiHeadAttention(node_dim, pair_dim, num_heads)
        self.norm2 = nn.LayerNorm(node_dim)
        self.ff = FeedForward(node_dim, multiplier=ff_mult)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # s shape: (B, L, node_dim), z shape: (B, L, L, pair_dim)
        
        # First sub-layer: Attention
        s = s + self.attn(self.norm1(s), z)
        
        # Second sub-layer: Feed-forward
        s = s + self.ff(self.norm2(s))
        
        return s
    

class InvariantEncoder(nn.Module):
    """
    This module orchestrates the entire process from backbone coordinates to a
    low-dimensional latent space representation.
    """
    def __init__(self, node_embed_dim: int = 64, pair_embed_dim: int = 32, num_blocks: int = 4, num_heads: int = 4, 
                 latent_dim: int = 2, num_gaussians_ca: int=128, num_gaussians_no: int=64, use_skips: bool=False):
        super().__init__()

        self.use_skips = use_skips
        
        # Initial projection layers
        self.project_1d_features = nn.Linear(6, node_embed_dim) # 6 for sin/cos of 3 angles
        
        # RBF modules for distances
        self.rbf_ca = GaussianSmearing(start=0.0, stop=20.0, num_gaussians=num_gaussians_ca)
        self.rbf_no = GaussianSmearing(start=0.0, stop=5.0, num_gaussians=num_gaussians_no)
        
        # Projection for combined distance features
        self.project_2d_features = nn.Linear(num_gaussians_ca + num_gaussians_no, pair_embed_dim)
        
        # Positional encoding
        self.positional_encoding = AF2_PositionalEmbedding(pair_embed_dim)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(node_dim=node_embed_dim, pair_dim=pair_embed_dim, num_heads=num_heads)
            for _ in range(num_blocks)
        ])
        
        # Final output layer to project to latent space
        self.output_projection = nn.Linear(node_embed_dim, latent_dim)

    def forward(self, batch_coords: torch.Tensor, residx: torch.Tensor) -> torch.Tensor:
        # batch_coords shape: (B, L, 4, 3)
        # residx shape: (B, L)
        
        # 1. Feature Extraction
        s_in = get_backbone_torsion_features(batch_coords)
        z_ca = get_distance_features(batch_coords[:, :, 1], batch_coords[:, :, 1], self.rbf_ca)
        z_no = get_distance_features(batch_coords[:, :, 0], batch_coords[:, :, 3], self.rbf_no)
        
        # 2. Initial Projections
        s = self.project_1d_features(s_in) # (B, L, node_embed_dim)
        z_dist = torch.cat([z_ca, z_no], dim=-1)
        z = self.project_2d_features(z_dist) # (B, L, L, pair_embed_dim)
        
        # 3. Add Positional Encoding
        z = z + self.positional_encoding(residx)

        if self.use_skips:
            skips = []
        
        # 4. Process through Transformer Blocks
        for block in self.blocks:
            s = block(s, z)
            # Only store the intermediate 's' if the flag is set
            skips.append(s)
        # 5. Get final latent encoding
        # We can average the residue embeddings to get a single vector per structure
        s_mean = s.mean(dim=1) # (B, node_embed_dim)
        latent_vec = self.output_projection(s_mean) # (B, latent_dim)
        
        # Return the full set of tensors for the U-Net decoder
        if self.use_skips:
            return latent_vec, z, skips
        else:
        # Return only the latent vector for an alternative decoder
            return latent_vec


class UNetDecoder(nn.Module):
    """
    Decodes a latent vector back to 3D coordinates using a U-Net style
    architecture with skip connections from an encoder.
    """
    def __init__(self, 
                 latent_dim: int, 
                 L: int, 
                 node_dim: int, 
                 pair_dim: int, 
                 num_blocks: int, 
                 num_heads: int):
        super().__init__()
        self.L = L
        self.node_dim = node_dim

        # 1. The initial "un-pooling" layer
        self.unpool_mlp = nn.Sequential(
            nn.Linear(latent_dim, L * node_dim),
            nn.GELU()
        )

        # 2. Decoder blocks. Note that the input dimension to the internal layers
        #    will be 2 * node_dim because of the skip connection concatenation.
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                node_dim=node_dim * 2, # Input is concatenated s_dec and s_skip
                pair_dim=pair_dim, 
                num_heads=num_heads
            ) for _ in range(num_blocks)
        ])
        
        # A projection layer to bridge the output of one block (node_dim*2) 
        # to the input of the next (node_dim) before concatenation.
        self.projection_layers = nn.ModuleList([
            nn.Linear(node_dim * 2, node_dim) for _ in range(num_blocks)
        ])

        # 3. Final projection layer to predict coordinates
        self.coord_projection = nn.Linear(node_dim * 2, 4 * 3) # 4 atoms, 3 coords each

    def forward(self, latent_vec: torch.Tensor, 
                skip_connections: list[torch.Tensor], 
                z_pair_bias: torch.Tensor) -> torch.Tensor:
        # latent_vec shape: (B, latent_dim)
        # skip_connections: A list of encoder outputs, from shallow to deep
        # z_pair_bias: The 2D pair representation from the encoder
        B = latent_vec.shape[0]

        # 1. Un-pool the latent vector to initialize the decoder's 1D state
        s = self.unpool_mlp(latent_vec).view(B, self.L, self.node_dim)

        # 2. Iteratively process through decoder blocks, applying skip connections
        #    in reverse order (from deep/abstract to shallow/detailed).
        for i in range(len(self.decoder_blocks)):
            # Get the skip connection from the corresponding encoder layer
            skip = skip_connections[-(i + 1)]
            
            # Concatenate the current decoder state with the skip connection
            s_cat = torch.cat([s, skip], dim=-1) # Shape: (B, L, 2 * node_dim)
            
            # Pass through the decoder block
            s_refined = self.decoder_blocks[i](s_cat, z_pair_bias)
            
            # Project back to the original node_dim for the next iteration's residual
            s = self.projection_layers[i](s_refined)

        # 3. Project final refined representation to coordinates
        #    We use the final concatenated state as it has the richest information
        final_s = torch.cat([s, skip_connections[0]], dim=-1)
        pred_coords = self.coord_projection(final_s).view(B, self.L, 4, 3)
        
        return pred_coords


class InvariantAutoencoder(nn.Module):
    """
    TODO.
    """
    def __init__(self, L: int, node_dim: int, pair_dim: int, latent_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        
        # Instantiate the full encoder
        self.encoder = InvariantEncoder(
            L=L,
            node_dim=node_dim,
            pair_dim=pair_dim,
            latent_dim=latent_dim,
            num_blocks=num_blocks,
            num_heads=num_heads
        )
        
        # Instantiate the full decoder
        self.decoder = UNetDecoder(
            latent_dim=latent_dim,
            L=L,
            node_dim=node_dim,
            pair_dim=pair_dim,
            num_blocks=num_blocks,
            num_heads=num_heads
        )

    def forward(self, batch_coords: torch.Tensor):
        # --- 1. ENCODE ---
        # The encoder must be modified to return the latent vector, the final pair bias 'z',
        # and the list of skip connections.
        latent_vec, z_pair_bias, skips = self.encoder(batch_coords)
        
        # --- 2. DECODE ---
        # Pass the encoder's outputs directly to the decoder
        reconstructed_coords = self.decoder(latent_vec, skips, z_pair_bias)
        
        return reconstructed_coords    


    
