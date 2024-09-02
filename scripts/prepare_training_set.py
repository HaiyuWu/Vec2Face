import os
from os import path, makedirs
import lmdb
import msgpack
import numpy as np
import pandas as pd
from PIL import Image
from os import path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class ImageListRaw(ImageFolder):
    def __init__(self, feature_list, label_file, image_list):
        image_names = np.asarray(pd.read_csv(image_list, delimiter=" ", header=None))
        feature_names = np.asarray(pd.read_csv(feature_list, delimiter=" ", header=None))
        self.im_samples = np.sort(image_names[:, 0])
        self.feat_samples = np.sort(feature_names[:, 0])

        self.targets = np.loadtxt(label_file, int)
        self.classnum = np.max(self.targets) + 1

        print(self.classnum)

    def __len__(self):
        return len(self.im_samples)

    def __getitem__(self, index):
        assert path.split(self.im_samples[index])[1][:-4] == path.split(self.feat_samples[index])[1][:-4]

        with open(self.im_samples[index], "rb") as f:
            img = f.read()
        with open(self.feat_samples[index], "rb") as f:
            feature = f.read()
        return img, feature, self.targets[index]


class CustomRawLoader(DataLoader):
    def __init__(self, workers, feature_list, label_file, image_list):
        self._dataset = ImageListRaw(feature_list, label_file, image_list)

        super(CustomRawLoader, self).__init__(
            self._dataset, num_workers=workers, collate_fn=lambda x: x
        )


def list2lmdb(
    feature_list,
    label_file,
    image_list,
    dest,
    file_name,
    num_workers=16,
    write_frequency=50000,
):
    print("Loading dataset from %s" % image_list)
    data_loader = CustomRawLoader(
        num_workers, feature_list, label_file, image_list
    )
    name = f"{file_name}.lmdb"
    if not path.exists(dest):
        makedirs(dest)
    lmdb_path = path.join(dest, name)
    isdir = path.isdir(lmdb_path)

    print(f"Generate LMDB to {lmdb_path}")

    image_size = 112
    size = len(data_loader.dataset) * image_size * image_size * 3
    print(f"LMDB max size: {size}")

    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=size * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(len(data_loader.dataset))
    txn = db.begin(write=True)
    for idx, data in tqdm(enumerate(data_loader)):
        image, feature, label = data[0]
        txn.put(
            "{}".format(idx).encode("ascii"), msgpack.dumps((image, feature, int(label)))
        )
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)
    idx += 1

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", msgpack.dumps(keys))
        txn.put(b"__len__", msgpack.dumps(len(keys)))
        txn.put(b"__classnum__", msgpack.dumps(int(data_loader.dataset.classnum)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list", "-im", help="List of images.")
    parser.add_argument("--feature_list", "-f", help="List of features.")
    parser.add_argument("--label_file", "-l", help="Identity label file.")
    parser.add_argument("--workers", "-w", help="Workers number.", default=8, type=int)
    parser.add_argument("--dest", "-d", help="Path to save the lmdb file.")
    parser.add_argument("--file_name", "-n", help="lmdb file name.")
    args = parser.parse_args()

    list2lmdb(
        args.feature_list,
        args.label_file,
        args.image_list,
        args.dest,
        args.file_name,
        args.workers,
    )
