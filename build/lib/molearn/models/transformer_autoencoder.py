import torch
import biobox as bb
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self):

    def forward(self, x):
        pass


class GaussianSmearing:
    def __init__(self):

    
    def forward(self, x):
        pass


class TransformerLayer(nn.Module):

    """Set up building blocks"""

    def __init__(self, 
                 in_dim,
                 d_model, 
                 nhead,
                 dp_attn_norm="d_model",
                 in_dim_2d=None,
                 use_bias_2d=True
                 ):
        """d_model = c*n_head"""

        super(TransformerLayer, self).__init__()

        """Calculate dimension of each attention head"""

        head_dim = d_model // nhead
        assert head_dim * nhead == d_model, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim
        self.in_dim_2d = in_dim_2d

        """Attention Normalization"""

        if dp_attn_norm not in ("d_model", "head_dim"):
            raise KeyError("Unkown 'dp_attn_norm': %s" % dp_attn_norm)
        self.dp_attn_norm = dp_attn_norm

        """Q, K, V Projections (learned during training)"""

        # Linear layers for q, k, v for dot product affinities.
        self.q_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.k_linear = nn.Linear(in_dim, self.d_model, bias=False)
        self.v_linear = nn.Linear(in_dim, self.d_model, bias=False)

        """Output projection from d_model to in_dim for layer stacking"""

        # Output layer.
        out_linear_in = self.d_model
        self.out_linear = nn.Linear(out_linear_in, in_dim)

        """Create small MLP for biasing based on 2D representation.
        Learbs a specific bias for each residue in each attention head"""

        # Branch for 2d representation.
        if self.in_dim_2d is not None:
            self.mlp_2d = nn.Sequential(# nn.Linear(in_dim_2d, in_dim_2d),
                                        # nn.ReLU(),
                                        # nn.Linear(in_dim_2d, in_dim_2d),
                                        # nn.ReLU(),
                                        nn.Linear(in_dim_2d, self.nhead,
                                                  bias=use_bias_2d))

    def forward(self, s, _k, _v, p):

        #----------------------
        # Prepare the  input. -
        #----------------------

        """s is the input protein sequence. Not yet sure what form this is in"""

        # Receives a (L, N, I) tensor.
        # L: sequence length,
        # N: batch size,
        # I: input embedding dimension.
        seq_l, b_size, _e_size = s.shape
        # Compute attention norm
        if self.dp_attn_norm == "d_model":
            w_t = 1/np.sqrt(self.d_model)
        elif self.dp_attn_norm == "head_dim":
            w_t = 1/np.sqrt(self.head_dim)
        else:
            raise KeyError(self.dp_attn_norm)

        #----------------------------------------------
        # Compute q, k, v for dot product affinities. -
        #----------------------------------------------

        """Searching for a book in a library. You have:
        - a search query (Q)
        - a list of keywords (K) for each book to compare Q to
        - the actual contents of each book (V)
        The model learns the best way to create Q, K, and V"""

        # Compute q, k, v vectors. Will reshape to (L, N, D*H).
        # D: number of dimensions per head,
        # H: number of head,
        # E = D*H: embedding dimension.
        q = self.q_linear(s)
        k = self.k_linear(s)
        v = self.v_linear(s)

        "Preparation for multi-head attention."
        "We might have one head for 'local structure'"
        "and another for 'long-range interactions'"

        "Reshaping for MHA. We group the data by 'head' instead of by batch item."
        "This allows for massive parallel computation across heads."

        # Actually compute dot prodcut affinities.
        # Reshape first to (N*H, L, D).
        q = q.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        q = q * w_t
        k = k.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(seq_l, b_size * self.nhead, self.head_dim).transpose(0, 1)

        """Compute the dot product to obtain relevance scores between q an k vectors."""

        # Then perform matrix multiplication between two batches of matrices.
        # (N*H, L, D) x (N*H, D, L) -> (N*H, L, L)
        dp_aff = torch.bmm(q, k.transpose(-2, -1))

        #--------------------------------
        # Compute the attention values. -
        #--------------------------------

        tot_aff = dp_aff


        "Pass 2D distance matrix through small network."
        "Add the result to bias relevance scores."
        "The network can learn ot output a large positive number for residues that are close in space"
        "Pay extra attention to this pair because I am telling you that they are structurally important."

        # Use the 2d branch.
        if self.in_dim_2d is not None:
            p = self.mlp_2d(p)
            # (N, L1, L2, H) -> (N, H, L2, L1)
            p = p.transpose(1, 3)
            # (N, H, L2, L1) -> (N, H, L1, L2)
            p = p.transpose(2, 3)
            # (N, H, L1, L2) -> (N*H, L1, L2)
            p = p.contiguous().view(b_size*self.nhead, seq_l, seq_l)
            tot_aff = tot_aff + p

        """Apply softmax to final scores.
        Each row should be positive and add to one."""

        attn = nn.functional.softmax(tot_aff, dim=-1)
        # if dropout_p > 0.0:
        #     attn = dropout(attn, p=dropout_p)

        #-----------------
        # Update values. -
        #-----------------

        """The representation for each residue becomes:
        a weighted sum of all v vectors in the sequence, based on attention probabilities.
        This creates a more context-aware representation."""

        # Update values obtained in the dot product affinity branch.
        s_new = torch.bmm(attn, v)

        """We need to get back to a single output vector for each amino acid.
        The final layer learns how best to stitch the results of each attention head back together.
        The final output can then be passed through the next TransformerLayer in the stack."""
        # Reshape the output, that has a shape of (N*H, L, D) back to (L, N, D*H).
        s_new = s_new.transpose(0, 1).contiguous().view(seq_l, b_size, self.d_model)

        # Compute the ouput.
        output = s_new
        output = self.out_linear(output)
        return (output, )

class TransformerBlock(nn.Module):


class TransformerEncoder(nn.Module):


class TransformerDecoder(nn.Module):


class TransformerAutoencoder(nn.Module):


