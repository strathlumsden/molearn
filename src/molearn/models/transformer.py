import math
import torch
import biobox as bb
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
        """New module documentation: TODO."""

    def __init__(self,
                 pos_embed_dim: int,
                 pos_embed_r: int = 32,
                 dim_order: str = "transformer"
                ):
        super().__init__()
        self.embed = nn.Embedding(pos_embed_r*2+1, pos_embed_dim)
        self.pos_embed_r = pos_embed_r
        self.set_dim_order(dim_order)

    def set_dim_order(self, dim_order):
        self.dim_order = dim_order
        if self.dim_order == "transformer":
            self.l_idx = 0  # Token (residue) index.
            self.b_idx = 1  # Batch index.
        elif self.dim_order == "trajectory":
            self.l_idx = 1
            self.b_idx = 0
        else:
            raise KeyError(dim_order)
        
    def forward(self, x, r=None):
        """
        x: xyz coordinate tensor of shape (L, B, *) if `dim_order` is set to
            'transformer'.
        r: optional, residue indices tensor of shape (B, L).

        returns:
        p: 2d positional embedding of shape (B, L, L, `pos_embed_dim`).
        """
        if r is None:
            prot_l = x.shape[self.l_idx]
            p = torch.arange(0, prot_l, device=x.device)
            p = p[None,:] - p[:,None]
            bins = torch.linspace(-self.pos_embed_r, self.pos_embed_r,
                                self.pos_embed_r*2+1, device=x.device)
            b = torch.argmin(
                torch.abs(bins.view(1, 1, -1) - p.view(p.shape[0], p.shape[1], 1)),
                axis=-1)
            p = self.embed(b)
            p = p.repeat(x.shape[self.b_idx], 1, 1, 1)
        else:
            b = r[:,None,:] - r[:,:,None]
            b = torch.clip(b, min=-self.pos_embed_r, max=self.pos_embed_r)
            b = b + self.pos_embed_r
            p = self.embed(b)
        return p   


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
    """Transformer layer block from idpGAN."""

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        num_heads: int,
        embed_2d_dim: int = None,
        norm_pos: str = "pre",
        activation: Callable = nn.ReLU,
        d_model: int = None,
        add_bias_kv: bool = True,
        embed_inject_mode: str = "adanorm",
        embed_2d_inject_mode: str = None,
        bead_embed_dim: int = 32,
        pos_embed_dim: int = 64,
        use_bias_2d: int = True,
        attention_type: str = "transformer"
        # input_inject_mode: str = None,
        # input_inject_pos: str = "out",
    ):

        ### Initialize and store the attributes.
        super().__init__()

        if d_model is None:
            d_model = embed_dim
        if not norm_pos in ("pre", "post"):
            raise KeyError(norm_pos)
        self.norm_pos = norm_pos
        self.attention_type = attention_type

        ### Transformer layer.
        # Edge features (2d embeddings).
        if embed_2d_dim is not None:
            if embed_2d_inject_mode == "add":
                attn_in_dim_2d = embed_2d_dim
                if embed_2d_dim != pos_embed_dim:
                    self.project_pos_embed_dim = nn.Linear(pos_embed_dim,
                                                           embed_2d_dim)
                else:
                    self.project_pos_embed_dim = nn.Identity()
            elif embed_2d_inject_mode == "concat":
                attn_in_dim_2d = embed_2d_dim + pos_embed_dim
            elif embed_2d_inject_mode is None:
                raise ValueError(
                    "Please provide a `embed_2d_inject_mode` when using"
                    " `embed_2d_dim` != 'None'")
            else:
                raise KeyError(embed_2d_inject_mode)
            self.embed_2d_inject_mode = embed_2d_inject_mode
        else:
            self.embed_2d_inject_mode = None
            attn_in_dim_2d = pos_embed_dim

        # Actual transformer layer.
        self.attn_norm = nn.LayerNorm(
            embed_dim,
            elementwise_affine=embed_inject_mode != "adanorm")
        if attention_type == "transformer":
            self.self_attn = TransformerLayerIdpGAN(
                in_dim=embed_dim,
                d_model=d_model,
                nhead=num_heads,
                dp_attn_norm="d_model",  # dp_attn_norm="head_dim",
                in_dim_2d=attn_in_dim_2d,
                use_bias_2d=use_bias_2d)
        elif attention_type  == "timewarp":
            self.self_attn = TransformerTimewarpLayer(
                in_dim=embed_dim,
                d_model=d_model,
                nhead=num_heads,
                in_dim_2d=attn_in_dim_2d,
                use_bias_2d=use_bias_2d)
        else:
            raise KeyError(attention_type)
        
        ### MLP.
        if embed_inject_mode is not None:
            if embed_inject_mode == "concat" and self.norm_pos == "post":
                # IdpGAN mode.
                fc1_in_dim = embed_dim + bead_embed_dim
            else:
                fc1_in_dim = embed_dim
        else:
            fc1_in_dim = embed_dim
        self.fc1 = nn.Linear(fc1_in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.final_norm = nn.LayerNorm(
            embed_dim,
            elementwise_affine=embed_inject_mode != "adanorm")
        self.act = activation()

        ### Conditional information injection module.
        self.cond_injection_module = AE_ConditionalInjectionModule(
            mode=embed_inject_mode,
            embed_dim=embed_dim,
            bead_embed_dim=bead_embed_dim,
            activation=activation,
            norm_pos=norm_pos
        )

    def forward(self, x, a, p, z=None, x_0=None):

        # Attention mechanism.
        residual = x
        inj_out = self.cond_injection_module(a=a)
        x = self.cond_injection_module.inject_0(x, inj_out)
        if self.norm_pos == "pre":
            x = self.attn_norm(x)
        x = self.cond_injection_module.inject_1_pre(x, inj_out)
        if self.embed_2d_inject_mode == "add":
            z_hat = z + self.project_pos_embed_dim(p)
        elif self.embed_2d_inject_mode == "concat":
            z_hat = torch.cat([z, p], axis=3)
        elif self.embed_2d_inject_mode is None:
            z_hat = p
        else:
            raise KeyError(self.embed_2d_inject_mode)
        x = self.self_attn(x, x, x, p=z_hat)[0]
        attn = None
        x = self.cond_injection_module.inject_1_post(x, inj_out)
        x = residual + x
        if self.norm_pos == "post":
            x = self.attn_norm(x)

        # MLP update.
        residual = x
        if self.norm_pos == "pre":
            x = self.final_norm(x)
        x = self.cond_injection_module.inject_2_pre(x, inj_out)
        x = self.fc2(self.act(self.fc1(x)))
        x = self.cond_injection_module.inject_2_post(x, inj_out)
        x = residual + x
        if self.norm_pos == "post":
            x = self.final_norm(x)

        # # Inject initial input.
        # x = self.inject_input(x, x_0, pos="out")

        return x, attn


class TransformerEncoder(nn.Module):


class TransformerDecoder(nn.Module):


class TransformerAutoencoder(nn.Module):


