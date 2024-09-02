import numpy as np
from collections import defaultdict
import argparse


def identity_collector(args):
    im_paths = np.sort(np.genfromtxt(args.image_path, str))
    id_dict = defaultdict(list)
    id_list = []
    for im_path in im_paths:
        im_id = im_path.split("/")[-2]
        id_dict[im_id].append(im_path)

    for i, (_, v) in enumerate(id_dict.items()):
        for _ in v:
            id_list.append(i)

    np.savetxt(f"{args.destination}/{args.name}_labels.txt", id_list, fmt="%d")

    # im_paths = np.genfromtxt(args.image_path, str)
    # id_dict = defaultdict(int)
    # for im_path in im_paths:
    #     im_id = im_path.split("/")[-2]
    #     id_dict[im_id] += 1
    # print(np.mean(list(id_dict.values())))
    # print(np.std(list(id_dict.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Collect identity labels."
    )
    parser.add_argument(
        "--image_path", "-im", help="A file that contains image paths.", type=str
    )
    parser.add_argument("--destination", "-d", help="destination.", type=str)
    parser.add_argument("--name", "-n", help="file name.", type=str)

    args = parser.parse_args()

    identity_collector(args)
