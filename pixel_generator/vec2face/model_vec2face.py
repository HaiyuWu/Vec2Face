from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp
from omegaconf import OmegaConf
import numpy as np
import scipy.stats as stats
from pixel_generator.vec2face.im_decoder import Decoder
from sixdrepnet.model import utils


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        with torch.cuda.amp.autocast(enabled=False):
            if return_attention:
                _, attn = self.attn(self.norm1(x))
                return attn
            else:
                y, _ = self.attn(self.norm1(x))
                x = x + self.drop_path(y)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
            self, input_ids
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_ids + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MaskedGenerativeEncoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=112, patch_size=7, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 use_rep=True, rep_dim=512,
                 rep_drop_prob=0.0,
                 use_class_label=False):
        super().__init__()
        assert not (use_rep and use_class_label)

        # --------------------------------------------------------------------------
        vqgan_config = OmegaConf.load('configs/vec2face/vqgan.yaml').model
        self.token_emb = BertEmbeddings(hidden_size=embed_dim,
                                        max_position_embeddings=49 + 1,
                                        dropout=0.1)
        self.use_rep = use_rep
        self.use_class_label = use_class_label
        if self.use_rep:
            print("Use representation as condition!")
            self.latent_prior_proj_f = nn.Linear(rep_dim, embed_dim, bias=True)
        # CFG config
        self.rep_drop_prob = rep_drop_prob
        self.feature_token = nn.Linear(1, 49, bias=True)
        self.center_token = nn.Linear(embed_dim, 49, bias=True)
        self.im_decoder = Decoder(**vqgan_config.params.ddconfig)
        self.im_decoder_proj = nn.Linear(embed_dim, vqgan_config.params.ddconfig.z_channels)

        # Vec2Face variant masking ratio
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)
        # --------------------------------------------------------------------------
        # Vec2Face encoder specifics
        dropout_rate = 0.1
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # Vec2Face decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.pad_with_cls_token = True

        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=True)  # learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.xavier_uniform_(self.feature_token.weight)
        torch.nn.init.xavier_uniform_(self.center_token.weight)
        torch.nn.init.xavier_uniform_(self.latent_prior_proj_f.weight)
        torch.nn.init.xavier_uniform_(self.decoder_embed.weight)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, rep):
        # expand to feature map
        device = rep.device
        encode_feature = self.latent_prior_proj_f(rep)
        feature_token = self.feature_token(encode_feature.unsqueeze(-1)).permute(0, 2, 1)

        gt_indices = torch.cat((encode_feature.unsqueeze(1), feature_token), dim=1).clone().detach()

        # masked row indices
        bsz, seq_len, _ = feature_token.size()
        mask_ratio_min = self.mask_ratio_min
        mask_rate = self.mask_ratio_generator.rvs(1)[0]

        num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))

        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        while True:
            noise = torch.rand(bsz, seq_len, device=rep.device)  # noise in [0, 1]
            sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
            cutoff_drop = sorted_noise[:, num_dropped_tokens - 1:num_dropped_tokens]
            cutoff_mask = sorted_noise[:, num_masked_tokens - 1:num_masked_tokens]
            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()
            if token_drop_mask.sum() == bsz * num_dropped_tokens and \
                    token_all_mask.sum() == bsz * num_masked_tokens:
                break
            else:
                print("Rerandom the noise!")
        token_all_mask_bool = token_all_mask.bool()
        encode_feature_expanded = encode_feature.unsqueeze(1).expand(-1, feature_token.shape[1], -1)
        feature_token[token_all_mask_bool] = encode_feature_expanded[token_all_mask_bool]

        # concatenate with image feature
        feature_token = torch.cat([encode_feature.unsqueeze(1), feature_token], dim=1)
        token_drop_mask = torch.cat([torch.zeros(feature_token.size(0), 1).to(device), token_drop_mask], dim=1)
        token_all_mask = torch.cat([torch.zeros(feature_token.size(0), 1).to(device), token_all_mask], dim=1)

        # bert embedding
        input_embeddings = self.token_emb(feature_token)

        bsz, seq_len, emb_dim = input_embeddings.shape

        # dropping
        token_keep_mask = 1 - token_drop_mask
        input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)

        # apply Transformer blocks
        x = input_embeddings_after_drop
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, gt_indices, token_drop_mask, token_all_mask

    def forward_decoder(self, x, token_drop_mask, token_all_mask):
        # embed incomplete feature map
        x = self.decoder_embed(x)
        # fill masked positions with image feature
        mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)
        # add pos embed
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        logits = self.decoder_norm(x)
        bsz, _, emb_dim = logits.shape
        # an image decoder
        decoder_proj = self.im_decoder_proj(logits[:, 1:, :].reshape(bsz, 7, 7, emb_dim)).permute(0, 3, 1, 2)
        return decoder_proj, logits

    def get_last_layer(self):
        return self.im_decoder.conv_out.weight

    def forward(self, rep):
        last_layer = self.get_last_layer()
        latent, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(rep)
        decoder_proj, logits = self.forward_decoder(latent, token_drop_mask, token_all_mask)
        image = self.im_decoder(decoder_proj)

        return gt_indices, logits, image, last_layer, token_all_mask

    def gen_image(self, rep, quality_model, fr_model, pose_model=None, age_model=None, class_rep=None,
                  num_iter=1, lr=1e-1, q_target=27, pose=60):
        rep_copy = rep.clone().detach().requires_grad_(True)
        optm = optim.Adam([rep_copy], lr=lr)

        i = 0
        while i < num_iter:
            latent, _, token_drop_mask, token_all_mask = self.forward_encoder(rep_copy)
            decoder_proj, _ = self.forward_decoder(latent, token_drop_mask, token_all_mask)
            image = self.im_decoder(decoder_proj).clip(max=1., min=-1.)
            # feature comparison
            out_feature = fr_model(image)
            if class_rep is None:
                id_loss = torch.mean(1 - torch.cosine_similarity(out_feature, rep))
            else:
                distance = 1 - torch.cosine_similarity(out_feature, class_rep)
                id_loss = torch.mean(torch.where(distance > 0.5, distance, torch.zeros_like(distance)))
            quality = quality_model(image)
            norm = torch.norm(quality, 2, 1, True)
            q_loss = torch.where(norm < q_target, q_target - norm, torch.zeros_like(norm))

            pose_loss = 0
            if pose_model is not None:
                # sixdrepnet
                bgr_img = image[:, [2, 1, 0], :, :]
                pose_info = pose_model(((bgr_img + 1) / 2))
                pose_info = utils.compute_euler_angles_from_rotation_matrices(
                    pose_info) * 180 / np.pi
                yaw_loss = torch.abs(pose - torch.abs(pose_info[:, 1].clip(min=-90, max=90)))
                pose_loss = torch.mean(yaw_loss)
            q_loss = torch.mean(q_loss)
            if pose_loss > 5 or id_loss > 0.4 or q_loss > 1:
                i -= 1
            loss = id_loss * 100 + q_loss + pose_loss
            optm.zero_grad()
            loss.backward(retain_graph=True)
            optm.step()
            i += 1

        latent, _, token_drop_mask, token_all_mask = self.forward_encoder(rep_copy)
        decoder_proj, _ = self.forward_decoder(latent, token_drop_mask, token_all_mask)
        image = self.im_decoder(decoder_proj).clip(max=1., min=-1.)

        return image, rep_copy.detach()


def vec2face_vit_base_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vec2face_vit_large_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vec2face_vit_huge_patch16(**kwargs):
    model = MaskedGenerativeEncoderViT(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=1280, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
