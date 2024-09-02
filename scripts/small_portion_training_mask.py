import numpy as np
from collections import defaultdict


if __name__ == '__main__':
    np.random.seed(0)
    percent = 0.1
    dataset = "WebFace4M"
    image_paths = np.sort(np.genfromtxt(f"./{dataset}.txt", str))
    info_with_id = defaultdict(list)
    for i, im_path in enumerate(image_paths):
        im_id = im_path.split("/")[-2]
        info_with_id[im_id].append(i)
    selected_ids = np.random.choice(list(info_with_id.keys()), 50000, replace=False)

    selected_im_pos = []
    for selected_id in selected_ids:
        selected_im_pos += info_with_id[selected_id]
    np.save(f"./small_portion_masks/{dataset}/{50000}_ids_{len(selected_im_pos)}_ims.npy",
            selected_im_pos)
