import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationEstimator(nn.Module):

    def __init__(self, channels=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, channels, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, 3, 1)
        )

    def forward(self, x):

        mean_c = x.mean(dim=1, keepdim=True)

        inp = torch.cat([x, mean_c], dim=1)

        illumination = torch.sigmoid(self.net(inp))

        return illumination


class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ReferenceEncoder(nn.Module):

    def __init__(self, base_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(

            ConvBlock(3, base_dim),

            ConvBlock(base_dim, base_dim * 2, stride=2),

            ConvBlock(base_dim * 2, base_dim * 4, stride=2),

            ConvBlock(base_dim * 4, base_dim * 4)
        )

    def forward(self, x):

        return self.encoder(x)


class MultiHeadCrossAttention(nn.Module):

    def __init__(self, dim, heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, query_feat, ref_feat):

        B, C, H, W = query_feat.shape

        query = query_feat.flatten(2).permute(0, 2, 1)

        key = ref_feat.flatten(2).permute(0, 2, 1)

        value = key

        attended, _ = self.attn(query, key, value)

        attended = self.norm(attended + query)

        attended = attended.permute(0, 2, 1).view(B, C, H, W)

        return attended


class TransformerBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            dim,
            4,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):

        B, C, H, W = x.shape

        feat = x.flatten(2).permute(0, 2, 1)

        attn_in = self.norm1(feat)

        attn_out, _ = self.attn(attn_in, attn_in, attn_in)

        feat = feat + attn_out

        ffn_in = self.norm2(feat)

        feat = feat + self.ffn(ffn_in)

        feat = feat.permute(0, 2, 1).view(B, C, H, W)

        return feat


class RetinexTransformer(nn.Module):

    def __init__(self, dim=256):
        super().__init__()

        self.blocks = nn.Sequential(
            TransformerBlock(dim),
            TransformerBlock(dim),
            TransformerBlock(dim)
        )

    def forward(self, x):

        return self.blocks(x)


class Decoder(nn.Module):

    def __init__(self, dim=256):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(dim, 128, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):

        return self.decoder(x)


class GANRefiner(nn.Module):

    def __init__(self):
        super().__init__()

        self.refine = nn.Sequential(

            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):

        return x + self.refine(x)


class RAGRetinexFormer(nn.Module):

    def __init__(self):
        super().__init__()

        self.illumination = IlluminationEstimator()

        self.low_encoder = ReferenceEncoder()

        self.ref_encoder = ReferenceEncoder()

        self.fusion = MultiHeadCrossAttention(
            dim=256
        )

        self.transformer = RetinexTransformer(
            dim=256
        )

        self.decoder = Decoder(
            dim=256
        )

        self.refiner = GANRefiner()

    def forward(self, low_img, ref_imgs):

        illumination_map = self.illumination(low_img)

        enhanced_input = low_img * illumination_map + low_img

        low_feat = self.low_encoder(enhanced_input)

        B, K, C, H, W = ref_imgs.shape

        ref_imgs = ref_imgs.view(B * K, C, H, W)

        ref_feat = self.ref_encoder(ref_imgs)

        ref_feat = ref_feat.view(
            B,
            K,
            ref_feat.shape[1],
            ref_feat.shape[2],
            ref_feat.shape[3]
        )

        ref_feat = torch.mean(ref_feat, dim=1)

        fused = self.fusion(
            low_feat,
            ref_feat
        )

        transformed = self.transformer(fused)

        decoded = self.decoder(transformed)

        refined = self.refiner(decoded)

        return torch.clamp(refined, -1, 1)


class PatchDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        def block(in_c, out_c, norm=True):

            layers = [
                nn.Conv2d(in_c, out_c, 4, 2, 1)
            ]

            if norm:
                layers.append(
                    nn.InstanceNorm2d(out_c)
                )

            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )

            return nn.Sequential(*layers)

        self.model = nn.Sequential(

            block(3, 64, norm=False),

            block(64, 128),

            block(128, 256),

            block(256, 512),

            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):

        return self.model(x)