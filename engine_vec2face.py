from typing import Iterable
import os
import torch
import util.misc as misc
from util.misc import get_grad_norm_
import util.lr_sched as lr_sched
import imageio
import numpy as np


def normalize_tensor(x):
    norm = torch.norm(x, 2, 1, True)
    output = torch.div(x, norm)
    return output


def save_images(gt_images, images, epoch, root):
    save_folder = f"{root}/epoch_{epoch}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for i, image in enumerate(images):
        imageio.imwrite(f"{save_folder}/original_{str(i).zfill(3)}.jpg", gt_images[i])
        imageio.imwrite(f"{save_folder}/{str(i).zfill(3)}.jpg", image)


def train_one_epoch(model: torch.nn.Module,
                    disc: torch.nn.Module,
                    data_loader: Iterable, loss: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, dis_optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler=None,
                    log_writer=None,
                    args=None):
    model.train(True)
    disc.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    optimizer.zero_grad()
    dis_optimizer.zero_grad()

    for data_iter_step, (gt_img, im_features, class_label) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_mage_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        if not args.use_amp:
            # original fp32
            gt_img = gt_img.to(device, non_blocking=True)
            im_features = im_features.to(device, non_blocking=True)
            gt_indices, logits, image, last_layer, *_ = model(im_features)
            ae_loss, d_loss, token_loss, rec_loss, ssim_loss, p_loss, feature_loss = loss(im_features,
                                                                                        gt_indices, logits,
                                                                                        gt_img, image, disc, 0,
                                                                                        epoch,
                                                                                        last_layer=last_layer)
            ae_loss_value = ae_loss.item()
            d_loss_value = d_loss.item()
            ae_loss = ae_loss / accum_iter
            d_loss = d_loss / accum_iter

            ae_loss.backward()
            d_loss.backward()

            # log grad norm
            if misc.get_rank() == 0:
                model_params = [p for p in model.parameters()]
                disc_params = [p for p in disc.parameters()]
                model_grad_norm = get_grad_norm_(model_params).item()
                disc_grad_norm = get_grad_norm_(disc_params).item()

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                dis_optimizer.step()
                dis_optimizer.zero_grad()

        else:
            # train with AMP fp16 or bf16
            with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=args.amp_dtype):
                gt_img = gt_img.to(device, non_blocking=True)
                im_features = im_features.to(device, non_blocking=True)
                gt_indices, logits, image, last_layer, *_ = model(im_features)
                ae_loss, d_loss, token_loss, rec_loss, ssim_loss, p_loss, feature_loss = loss(im_features,
                                                                                            gt_indices, logits,
                                                                                            gt_img, image, disc,
                                                                                            torch.tensor(0),
                                                                                            epoch,
                                                                                            last_layer=last_layer)
            ae_loss_value = ae_loss.item()
            d_loss_value = d_loss.item()
            ae_loss = ae_loss / accum_iter
            d_loss = d_loss / accum_iter

            if (data_iter_step + 1) % accum_iter == 0:
                model_grad_norm, disc_grad_norm = loss_scaler(ae_loss, d_loss, optimizer, dis_optimizer,
                            clip_grad=args.clip_grad, ae_parameters=model.parameters(), d_parameters=disc.parameters(),
                            update_grad=True)
                optimizer.zero_grad()
                dis_optimizer.zero_grad()
            else:
                model_grad_norm, disc_grad_norm = loss_scaler(ae_loss, d_loss, optimizer, dis_optimizer,
                            clip_grad=args.clip_grad, ae_parameters=model.parameters(), d_parameters=disc.parameters(),
                            update_grad=False)

        torch.cuda.synchronize()

        metric_logger.update(ae_loss=ae_loss_value)
        metric_logger.update(token_loss=token_loss)
        metric_logger.update(rec_loss=torch.mean(rec_loss))
        metric_logger.update(p_loss=torch.mean(p_loss))
        metric_logger.update(ssim_loss=ssim_loss)
        metric_logger.update(feature_loss=feature_loss)
        metric_logger.update(d_loss=d_loss_value)
        metric_logger.update(model_grad_norm=model_grad_norm)
        metric_logger.update(disc_grad_norm=disc_grad_norm)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        ae_loss_value_reduce = misc.all_reduce_mean(ae_loss_value)
        disc_loss_value_reduce = misc.all_reduce_mean(d_loss_value)
        token_loss_value_reduce = misc.all_reduce_mean(token_loss.item())
        rec_loss_value_reduce = misc.all_reduce_mean(torch.mean(rec_loss).item())
        ssim_loss_value_reduce = misc.all_reduce_mean(ssim_loss.item())
        p_loss_value_reduce = misc.all_reduce_mean(torch.mean(p_loss).item())
        feature_loss_value_reduce = misc.all_reduce_mean(feature_loss.item())
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('ae_train_loss', ae_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('disc_train_loss', disc_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('token_train_loss', token_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('rec_train_loss', rec_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('ssim_train_loss', ssim_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('p_train_loss', p_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('feature_train_loss', feature_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            if misc.get_rank() == 0:
                log_writer.add_scalar('model_grad_norm', model_grad_norm, epoch_1000x)
                log_writer.add_scalar('disc_grad_norm', disc_grad_norm, epoch_1000x)

    # gather the stats from all processes
    if misc.get_rank() == 0:
        save_images(((gt_img.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8),
                    ((image.permute(0, 2, 3, 1).detach().cpu().to(torch.float32).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8),
                    epoch, args.output_dir)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
