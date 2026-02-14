import torch
import torch.nn.functional as F
from models.vit import Attention

class SimpleTransformerBlock(torch.nn.Module):
    """Basic transformer block for decoder"""
    def __init__(self, dim=1024, num_heads=16, mlp_ratio=4.0, dropout=0.0, use_torch_attn=True):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.use_torch_attn = use_torch_attn
        if use_torch_attn:
            self.attn = torch.nn.MultiheadAttention(dim, num_heads, dropout=dropout, bias=True, batch_first=True)
        else:
            self.attn = Attention(dim=dim, num_heads=num_heads, attn_drop=dropout)
        self.norm2 = torch.nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_hidden_dim, dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        if self.use_torch_attn:
            attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        else:
            attn_output = self.attn(x_norm)
        x = x + attn_output

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViTDecoder(torch.nn.Module):
    """
    ViT Decoder that inverts LayerNorm normalization on raw pixel patches
    """
    def __init__(
        self,
        encoder_dim=1024,        # Input encoder dimension (raw pixel patches)
        decoder_dim=1024,       # Decoder hidden dimension
        num_layers=24,          # Number of transformer layers
        num_heads=16,           # Attention heads
        mlp_ratio=4.0,          # MLP expansion ratio
        target_resolution=256,  # Output image resolution
        patch_size=16,          # Output patch size
        dropout=0.0,             # Dropout
        output_channels=3,       # Output channels
        do_tanh=True,
        use_torch_attn=True
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size

        # Support both single value (square) and tuple (height, width)
        if isinstance(target_resolution, (list, tuple)):
            self.target_height, self.target_width = target_resolution
        else:
            self.target_height = self.target_width = target_resolution

        # Divide by patch size to get number of patches
        self.target_height = self.target_height // patch_size
        self.target_width = self.target_width // patch_size
        self.output_tokens = self.target_height * self.target_width
        self.output_channels = output_channels

        # 1. Project encoder features to decoder dimension
        self.input_proj = torch.nn.Linear(encoder_dim, decoder_dim)

        # 2. Learnable positional embeddings for OUTPUT space
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.output_tokens, decoder_dim))

        # 3. Transformer blocks (self-attention only)
        self.blocks = torch.nn.ModuleList([
            SimpleTransformerBlock(decoder_dim, num_heads, mlp_ratio, dropout, use_torch_attn=use_torch_attn)
            for _ in range(num_layers)
        ])

        # 4. Output projection to patches
        self.norm = torch.nn.LayerNorm(decoder_dim)
        if do_tanh:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(decoder_dim, patch_size * patch_size * self.output_channels),
                torch.nn.Tanh()  # Tanh output
            )
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(decoder_dim, patch_size * patch_size * self.output_channels)
            )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following ViT conventions"""
        # Position embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Linear layers
        def _init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)

    def interpolate_features(self, x, input_tokens=None, target_tokens=None):
        """Handle different encoder/decoder token counts"""
        batch_size, num_encoder_tokens, dim = x.shape

        # Use the model's pretrained input res
        # For now we assume inputs and outputs are same resolution, but this doesn't have to be the case
        if input_tokens is None:
            input_tokens = [self.target_height, self.target_width]
        if target_tokens is None:
            target_tokens_per_side = int(self.output_tokens ** 0.5)
            target_tokens = [self.target_height, self.target_width]

        if num_encoder_tokens == target_tokens or num_encoder_tokens == target_tokens[0] * target_tokens[1]:
            return x

        # Reshape to spatial, interpolate, reshape back
        x = x.transpose(1, 2)  # (B, dim, num_tokens)
        x = x.reshape(batch_size, dim, input_tokens[0], input_tokens[1])

        x = F.interpolate(x, size=(target_tokens[0], target_tokens[1]), mode='bilinear', align_corners=False)
        x = x.reshape(batch_size, dim, -1).transpose(1, 2)  # Back to (B, num_tokens, dim)

        return x

    def interpolate_pos_embed(self, target_h, target_w):
        new_pos_embed = self.pos_embed.reshape(
            1,
            self.target_height // self.patch_size,
            self.target_width // self.patch_size,
            self.decoder_dim
        ).permute(0, 3, 1, 2) # 1 x D x H' x W'

        new_pos_embed = F.interpolate(new_pos_embed, size=(target_h, target_w), mode='bilinear')
        # Reshape back
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).flatten(1,2)

        return new_pos_embed

    def load_decoder(self, decoder_path):
        """Load trained decoder"""

        checkpoint = torch.load(decoder_path, weights_only=False, map_location="cpu")

        # Handle DDP prefix
        decoder_state = checkpoint['decoder_state']
        if any(key.startswith('module.') for key in decoder_state.keys()):
            decoder_state = {key.replace('module.', '', 1): value for key, value in decoder_state.items()}

        msg = self.load_state_dict(decoder_state)
        print(msg)
        self.to("cuda").eval()

        # 🔍 DEBUG: Verify decoder dtype before matching
        decoder_dtype = next(self.parameters()).dtype
        print(f"✅ Decoder loaded with dtype: {decoder_dtype}")


    # hw_input specifies the grid structure of the encoder_features which are already flattened
    def forward(self, encoder_features, hw_input):
        """
        Forward pass
        Args:
            encoder_features: (batch, num_patches, encoder_dim) normalized raw pixel patches
        Returns:
            reconstructed_images: (batch, 3, target_resolution, target_resolution) in range [-1, 1]
        """
        from einops import rearrange

        # Reshape if there is time dimension
        B, T, N, D = encoder_features.shape
        encoder_features = encoder_features.reshape(B * T, N, D)

        # 1. Project encoder features to decoder dimension
        x = self.input_proj(encoder_features)  # (B, num_patches, decoder_dim)

        # 2. Interpolate to pretrained dimension of decoder
        # Reshapes x to hw_input grid, then interpolates to decoder dimension
        x = self.interpolate_features(x, input_tokens=hw_input)  # (B, output_tokens, decoder_dim)

        # Interpolate position embedding (Don't do it, better to run decoder in the original resolution it was trained at)
        interpolated_pos_embed = self.pos_embed
        # if hw_input[0] * hw_input[1] != interpolated_pos_embed.shape[1]:
        #     interpolated_pos_embed = self.interpolate_pos_embed(hw_input[0], hw_input[1])

        # 3. Add positional embeddings for OUTPUT space
        x = x + interpolated_pos_embed

        # 4. Self-attention processing
        for block in self.blocks:
            x = block(x)

        # 5. Output projection
        x = self.norm(x)
        x = self.head(x)  # (B, output_tokens, patch_size² * 3)

        # Interpolate the feature output to the expected size
        # x = self.interpolate_features(x, input_tokens=None, target_tokens=hw_input)  # (B, output_tokens, decoder_dim)

        # 6. Reshape patches to image
        # Reshape so that we can apply interpolation
        x = rearrange(
            x, '(b t) (h w) (p1 p2 c) -> (b t) c (h p1) (w p2)',
            t=T,
            h=self.target_height,
            w=self.target_width,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.output_channels
        )

        x = F.interpolate(x, (hw_input[0] * self.patch_size, hw_input[1] * self.patch_size), mode='bilinear')

        # Reshape time dimension out of batch dimension
        x = rearrange(
            x, '(b t) c (h p1) (w p2) -> b t c (h p1) (w p2)',
            t=T,
            h=hw_input[0],
            w=hw_input[1],
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.output_channels
        )

        return x

if __name__ == '__main__':
    decoder= SimpleViTDecoder(target_resolution=[224, 392], patch_size=14).cuda()
    decoder.load_decoder('decoder/vit_l_dinov2_vitl_nonorm_res224x392_robocasa_4node/checkpoints/checkpoint_0037500/checkpoint.pt')
    hw = [16, 28]
    input = torch.rand(1, 4, hw[0] * hw[1], 1024).cuda()

    output = decoder(input, hw)
    print(output.shape)
