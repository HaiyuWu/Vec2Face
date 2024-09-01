from os import path
import lmdb
import msgpack
import numpy as np
import six
import io
import torch
from tqdm import tqdm
from PIL import Image
import queue as Queue
import threading
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from torchvision import transforms
from .dist import DistributedSampler, get_dist_info


class LMDB(Dataset):
    def __init__(self, db_path, transform=None, mask=None):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            subdir=path.isdir(db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self.length = msgpack.loads(txn.get(b"__len__"))
            self.keys = msgpack.loads(txn.get(b"__keys__"))
            self.classnum = msgpack.loads(txn.get(b"__classnum__"))

        self.mask = None
        if mask is not None:
            self.mask = np.load(mask)
            self.length = len(self.mask)

        self.transform = transform

    def __getitem__(self, index):
        env = self.env
        if self.mask is not None:
            index = self.mask[index]
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = msgpack.loads(byteflow)

        # load image
        img = Image.open(io.BytesIO(unpacked[0])).convert("RGB")

        # load feature
        feature = torch.tensor(np.load(io.BytesIO(unpacked[1]), allow_pickle=True))
        # map to (mean=0 std=1) based on extracted features
        # WebFace - mean: 0.0046 std: 0.96731
        # MS1MV2 - TBD (Glint360K)

        # load label
        target = unpacked[2]

        if self.transform is not None:
            img = self.transform(img)

        return img, feature, target

    def __len__(self):
        return self.length


class LMDBDataLoader(object):
    def __init__(self, args, train=True, seed=2048, transform=None):
        transform = transforms.Compose(
            [
                transforms.Resize(112, interpolation=3),
                transforms.ToTensor()
            ]
        ) if transform is None else transform

        self._dataset = LMDB(args.train_source, transform, args.mask)
        rank, world_size = args.gpu, args.world_size
        samplers = DistributedSampler(self._dataset, num_replicas=world_size, rank=rank, shuffle=train, seed=seed)

        # use DataLoaderX for faster loading
        self.loader = DataLoaderX(
            local_rank=rank,
            dataset=self._dataset,
            batch_size=args.batch_size,
            sampler=samplers,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            drop_last=train,
        )

    def get_loader(self):
        return self.loader


#################################################################################################
# https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/dataset.py#L27
#################################################################################################
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=8):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.local_rank = local_rank
        self.stream = torch.cuda.Stream(self.local_rank)

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
