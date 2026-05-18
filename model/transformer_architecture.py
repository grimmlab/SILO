import torch
from torch import nn
from config import SequenceConfig
from model.mha import MHA

class SequenceTransformer(nn.Module):

    """
        Two level sequence design policy
        Supports multi-level actions:
        - Level 0: pick position
            [(0....L-1)]

        - Level 1: pick residue to insert at selected position
            {20 amino-acid logits for each candidate position}: [A(0), R(1), Y(2), N(3),...D(19)]

        Inputs (x):
            x["embeds"] : (B, L); embeddings with layout:
                            [pos1(0), pos2(1), posL(L-1)]
 
            x["valid_positions"]: a valid position mask

        Returns: 
            level_zero_logits: (B, L) seq length 
            level_one_logits: (B, L, 20)

    
    """
    
    def __init__(self, config: SequenceConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        
        self.latent_dim = config.latent_dimension
        self.num_heads = config.num_heads
        self.num_total_residues = len(config.residue_vocabulary) # all residues in vocab  
        self.linear_proj = nn.Linear(self.config.esm_emb_size, self.latent_dim)

        # Transformer encoders
        self.encoders = nn.ModuleList([
            TransformerEncoderWithFlashAttention(config = config, d_model = self.latent_dim, nhead = self.num_heads, dropout=config.dropout)
            for _ in range(config.num_transformer_blocks)
            ])

        #----heads----#
        self.level0_head = nn.Linear(self.latent_dim, 1) # Per-position logit 
        self.level1_head = nn.Linear(self.latent_dim, self.num_total_residues) # Per-position 20 AA logits for Level 1


    def forward(self, x: dict):
        
        latent_seq_embed = self.linear_proj(x["embeds"])

        for _, block in enumerate(self.encoders):
            latent_seq_embed = block(latent_seq_embed)

        #------Level 0: Select position------#
        level_zero_logits = self.level0_head(latent_seq_embed[:, 0:, :]).squeeze(-1) #(B, L, 1)
        level_zero_logits = level_zero_logits.masked_fill(~x["valid_positions"], -1e9) # mask out all non valid positions 
        
        #------Level 1: AA selection for every position------#

        level_one_logits = self.level1_head(latent_seq_embed[:, 0:, :])  
        level_one_logits = level_one_logits.masked_fill(~x["valid_positions"].unsqueeze(-1), -1e9)

        return level_zero_logits, level_one_logits #(B, L), #(B, L, 20)
        
    def get_weights(self):
        return dict_to_cpu(self.state_dict())


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict

class TransformerEncoderWithFlashAttention(nn.Module):
    def __init__(self, config: SequenceConfig, d_model, nhead, dropout):
        super(TransformerEncoderWithFlashAttention, self).__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.config = config
        self.attn = MHA(embed_dim=d_model, num_heads=nhead, use_flash_attn=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model), #4 *d_model, expansion factor
            nn.Dropout(dropout),
        )


    def forward(self, x):

        x_norm = self.layer_norm1(x)
        attn_outputs = self.attn(x_norm)
        x = x + attn_outputs # Do residuals (x + z)
        #FFN block
        ff_o = self.ffn(self.layer_norm2(x))
        x = ff_o + x # Do residuals (ff_o + h) and then apply layernorm
        return x

