import torch
import torch.nn as nn
import torch.nn.functional as F
from models import iresnet
from lpips.lpips import LPIPS
from pytorch_msssim import SSIM


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def mse_d_loss(logits_real, logits_fake):
    loss_real = torch.mean((logits_real - 1.) ** 2)
    loss_fake = torch.mean(logits_fake ** 2)
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real)) +
            torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def create_fr_model(model_path, depth="100", use_amp=True):
    model = iresnet(depth)
    model.load_state_dict(torch.load(model_path))
    if use_amp:
        model.half()
    return model


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start=1000, disc_factor=1.0, disc_weight=1.0,
                 disc_conditional=False, disc_loss="mse", id_loss="mse",
                 fr_model="./weights/arcface-r100-glint360k.pth",
                 not_use_g_loss_adaptive_weight=False, use_amp=True):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "mse", "smooth"]
        self.loss_name = disc_loss
        self.perceptual_loss = LPIPS().eval()
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "mse":
            self.disc_loss = mse_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.fr_model = create_fr_model(fr_model, use_amp=use_amp).eval()
        if id_loss == "mse":
            self.feature_loss = nn.MSELoss()
        elif id_loss == "cosine":
            self.feature_loss = nn.CosineSimilarity()
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.ssim_loss = SSIM(data_range=1, size_average=True, channel=3)
        self.not_use_g_loss_adaptive_weight = not_use_g_loss_adaptive_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, im_features, gt_indices, logits, gt_img, image, discriminator, emb_loss,
                epoch, last_layer=None, cond=None, mask=None):
        rec_loss = (image - gt_img) ** 2
        if epoch >= 0:
            gen_feature = self.fr_model(image)
            feature_loss = torch.mean(1 - torch.cosine_similarity(im_features, gen_feature))
        else:
            feature_loss = 0

        p_loss = self.perceptual_loss(image, gt_img) * 2

        with torch.cuda.amp.autocast(enabled=False):
            ssim_loss = 1 - self.ssim_loss((image.float() + 1) / 2, (gt_img + 1) / 2)
        logits_fake = discriminator(image)
        logits_real_d = discriminator(gt_img.detach())
        logits_fake_d = discriminator(image.detach())

        if mask is None:
            token_loss = (logits[:, 1:, :] - gt_indices[:, 1:, :])
            token_loss = torch.mean(token_loss ** 2)
        else:
            token_loss = torch.abs((logits[:, 1:, :] - gt_indices[:, 1:, :])) * mask[:, 1:, None]
            token_loss = token_loss.sum() / mask[:, 1:].sum()

        nll_loss = torch.mean(rec_loss + 0.1 * p_loss) + \
                   0.1 * ssim_loss + \
                   token_loss + feature_loss + emb_loss
        # generator update
        g_loss = -torch.mean(logits_fake)

        if not self.not_use_g_loss_adaptive_weight or epoch < self.discriminator_iter_start:
            d_weight = 1.0
        else:
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
        disc_factor = adopt_weight(self.disc_factor, epoch, threshold=self.discriminator_iter_start)

        ae_loss = nll_loss + d_weight * disc_factor * g_loss

        # second pass for discriminator update
        disc_factor = adopt_weight(self.disc_factor, epoch, threshold=self.discriminator_iter_start)
        d_loss = disc_factor * self.disc_loss(logits_real_d, logits_fake_d)
        return ae_loss, d_loss, token_loss, rec_loss, ssim_loss, p_loss, feature_loss
