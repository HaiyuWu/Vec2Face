import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from dataloader import LMDBDataLoader
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from pixel_generator.vec2face.taming.modules.discriminator_loss import VQLPIPSWithDiscriminator
from pixel_generator.vec2face.taming.modules.discriminator.model import Discriminator
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import pixel_generator.vec2face.model_vec2face as model_vec2face
from engine_vec2face import train_one_epoch

torch.autograd.set_detect_anomaly(True)


def get_args_parser():
    parser = argparse.ArgumentParser('vec2face training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vec2face_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=112, type=int,
                        help='images input size')

    # Pre-trained enc parameters
    parser.add_argument('--use_rep', action='store_true', help='use representation as condition.')
    parser.add_argument('--use_class_label', action='store_true', help='use class label as condition.')
    parser.add_argument('--rep_dim', default=512, type=int)

    # Pixel generation parameters
    parser.add_argument('--rep_drop_prob', default=0.0, type=float)

    # Vec2Face params
    parser.add_argument('--mask_ratio_min', type=float, default=0.5,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.75,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_source', type=str,
                        help='image path --- .lmdb file')
    parser.add_argument('--mask', default=None, help='training position mask', type=str)
    parser.add_argument('--augmentation', default='noaug', type=str,
                        help='Augmentation type')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--pin_memory', action='store_false',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # init log writer
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # simple augmentation
    if args.augmentation == "noaug":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])])
    elif args.augmentation == "randcrop":
        transform_train = transforms.Compose([
            transforms.Resize(112, interpolation=3),
            transforms.RandomCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    elif args.augmentation == "randresizedcrop":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    else:
        raise NotImplementedError

    dataset = LMDBDataLoader(args, transform=transform_train)
    data_loader_train = dataset.get_loader()
    # define the model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 112

    model = model_vec2face.__dict__[args.model](mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
                                                mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
                                                use_rep=args.use_rep,
                                                rep_dim=args.rep_dim,
                                                rep_drop_prob=args.rep_drop_prob,
                                                use_class_label=args.use_class_label)
    loss = VQLPIPSWithDiscriminator(disc_start=1000, disc_weight=0.8)
    disc = Discriminator(dims=(64, 128, 256, 512))

    model.to(device)
    disc.to(device)
    loss.to(device)

    model_without_ddp = model
    disc_without_ddp = disc
    print("Mage Model = %s" % str(model_without_ddp))
    print("VQ-disc Model = %s" % str(disc))

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

        disc = torch.nn.parallel.DistributedDataParallel(disc,
                                                         device_ids=[args.gpu],
                                                         find_unused_parameters=True)
        disc_without_ddp = disc.module

    # Log parameters
    n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    disc_n_params = sum(p.numel() for p in disc_without_ddp.parameters() if p.requires_grad)
    print("Number of trainable parameters for Vec2Face: {}M".format(n_params / 1e6))
    print("Number of trainable parameters for VQ disc: {}M".format(disc_n_params / 1e6))

    if global_rank == 0:
        log_writer.add_scalar('mage_num_params', n_params / 1e6, 0)
        log_writer.add_scalar('vq_disc_num_params', disc_n_params / 1e6, 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    dis_optimizer = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.9, 0.99))

    loss_scaler = NativeScaler()

    misc.load_model(args=args,
                    model_without_ddp=model_without_ddp,
                    disc_without_ddp=disc_without_ddp,
                    optimizer=optimizer,
                    disc_optimizer=dis_optimizer,
                    loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, disc, data_loader_train, loss,
            optimizer, dis_optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if train_stats is None:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                disc_without_ddp=disc_without_ddp, optimizer=optimizer, disc_optimizer=dis_optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if args.output_dir and (epoch % 25 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                disc_without_ddp=disc_without_ddp, optimizer=optimizer, disc_optimizer=dis_optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        misc.save_model_last(
            args=args, model=model, model_without_ddp=model_without_ddp,
            disc_without_ddp=disc_without_ddp, optimizer=optimizer, disc_optimizer=dis_optimizer,
            loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
