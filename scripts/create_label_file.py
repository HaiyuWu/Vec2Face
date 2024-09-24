import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
from os import path, makedirs


def get_labels(im_paths):
    id_dict = defaultdict(int)
    for im in tqdm(im_paths):
        identity = im.split("/")[-2]
        id_dict[identity] += 1
    labels = []
    for i, (k, value) in enumerate(id_dict.items()):
        labels += [i] * value
    return labels


def main(args):
    image_file = np.genfromtxt(args.image_file, str)
    labels = get_labels(image_file)
    makedirs(args.destination, exist_ok=True)
    np.savetxt(path.join(args.destination, f"{args.name}_labels.txt"), labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple example of creating label files.")
    parser.add_argument('--image_file', '-im', help='Path of the .txt file')
    parser.add_argument('--destination', '-d', help='Path of the saved folder')
    parser.add_argument('--name', '-n', help='name of the saved file')
    args = parser.parse_args()
    main(args)
