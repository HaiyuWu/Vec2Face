import torch
import argparse
import pixel_generator.vec2face.model_vec2face as model_vec2face
import imageio
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import os
from models import iresnet


def get_args_parser():
    parser = argparse.ArgumentParser('Vec2Face verify', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # Model parameters
    parser.add_argument('--model', default='vec2face_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=112, type=int,
                        help='images input size')

    # Pre-trained enc parameters
    parser.add_argument('--use_rep', action='store_false', help='use representation as condition.')
    parser.add_argument('--use_class_label', action='store_true', help='use class label as condition.')
    parser.add_argument('--rep_dim', default=512, type=int)

    # Pixel generation parameters
    parser.add_argument('--rep_drop_prob', default=0.0, type=float)
    parser.add_argument('--feat_batch_size', default=2000, type=int)
    parser.add_argument('--example', default=50, type=int)
    parser.add_argument('--center_feature', default=None, type=str)
    parser.add_argument('--name', default=None, type=str)

    # Vec2Face params
    parser.add_argument('--mask_ratio_min', type=float, default=0.1,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.15,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')
    parser.add_argument('--model_weights', default='',
                        help='model weights')
    parser.add_argument('--start_end', default=None,
                        help='slicing dataset generation')
    return parser.parse_args()


def save_images(images, id_num, root, name):
    global j, prev_id
    save_root = f"{root}/{name}"
    for i, image in enumerate(images):
        save_folder = f"{save_root}/{id_num[i]:04d}"
        os.makedirs(save_folder, exist_ok=True)
        if prev_id != id_num[i]:
            prev_id = id_num[i]
            j = 0
        imageio.imwrite(f"{save_folder}/{str(j).zfill(3)}.jpg", image)
        j += 1


def easy_to_generate_id(sampled_features):
    id_features = torch.tensor(np.load(sampled_features))
    return id_features


def sample_nearby_vectors(base_vector, epsilons, percentages=[0.4, 0.4, 0.2]):
    row, col = base_vector.shape
    norm = torch.norm(base_vector, 2, 1, True)
    diff = []
    for i, eps in enumerate(epsilons):
        diff.append(np.random.normal(0, eps, (int(row * percentages[i]), col)))
    diff = np.vstack(diff)
    np.random.shuffle(diff)
    diff = torch.tensor(diff)
    print(diff.shape)
    generated_samples = base_vector + diff
    generated_samples = generated_samples / torch.norm(generated_samples, 2, 1, True) * norm
    return generated_samples


def _create_fr_model(model_path="./weights/magface-r100-glint360k.pth", depth="100"):
    model = iresnet(depth)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == '__main__':
    args = get_args_parser()
    j = 0
    prev_id = -1

    dim = args.rep_dim
    batch_size = args.feat_batch_size
    example = args.example
    center_feature_file = args.center_feature
    name = args.name

    print("Loading model...")
    device = torch.device('cuda')
    model = model_vec2face.__dict__[args.model](mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
                                                mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
                                                use_rep=args.use_rep,
                                                rep_dim=args.rep_dim,
                                                rep_drop_prob=args.rep_drop_prob,
                                                use_class_label=args.use_class_label)

    model = model.to(device)
    checkpoint = torch.load(args.model_weights, map_location='cuda')
    model.load_state_dict(checkpoint['model_vec2face'])
    model.eval()

    if args.start_end is not None:
        start, end = args.start_end.split(":")
        assert int(end) > int(start)

    print("Loading estimators...")
    # quality model
    scorer = _create_fr_model().to(device)
    # id model
    fr_model = _create_fr_model("./weights/arcface-r100-glint360k.pth").to(device)

    id_pool = []
    bs_factor = 1
    print(f"Use ID vectors from: {center_feature_file}")
    random_ids = easy_to_generate_id(center_feature_file)

    random_ids = torch.tensor(random_ids)[int(start):int(end)].to(torch.float32)
    raw_cls_label = torch.arange(0, random_ids.shape[0])

    updated_feature = torch.empty(random_ids.shape)

    for i in tqdm(range(0, len(random_ids), args.batch_size * bs_factor)):
        im_features = random_ids[i: i + args.batch_size * bs_factor].to(device, non_blocking=True)
        images, im_features = model.gen_image(im_features, scorer, fr_model, class_rep=im_features)
        updated_feature[i: i + args.batch_size * bs_factor] = im_features

    class_label = torch.repeat_interleave(raw_cls_label, example, dim=0)
    expanded_ids = torch.repeat_interleave(updated_feature, example, dim=0).to(torch.float32)
    samples = sample_nearby_vectors(expanded_ids,
                                    epsilons=[0.3, 0.5, 0.7],
                                    percentages=[0.4, 0.4, 0.2]).to(torch.float32)
    samples = samples.to(device, non_blocking=True)

    collection = []
    print("start generating...")
    for i in tqdm(range(0, len(expanded_ids), args.batch_size)):
        im_features = samples[i: i + args.batch_size]
        _, _, image, *_ = model(im_features)
        save_images(((image.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8),
                    class_label[i: i + args.batch_size],
                    "generated_images",
                    name)
