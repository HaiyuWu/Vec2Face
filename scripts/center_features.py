import numpy as np
from tqdm import tqdm
import argparse
from os import path, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from collections import OrderedDict

def load_and_process_feature(feature_path):
    return feature_path.split("/")[-2], np.load(feature_path)

def process_chunk(chunk):
    return [load_and_process_feature(feature) for feature in chunk]

def process_features(features, num_cpus):
    chunk_size = max(10000, len(features) // (num_cpus * 2))  # Ensure at least 2 chunks per CPU
    chunks = [features[i:i + chunk_size] for i in range(0, len(features), chunk_size)]

    processed_features = []
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(chunks), desc="Processing chunks"):
            processed_features.extend(future.result())

    # Use OrderedDict to maintain order and group features by image ID
    grouped_features = OrderedDict()
    for img_id, feature in processed_features:
        if img_id not in grouped_features:
            grouped_features[img_id] = []
        grouped_features[img_id].append(feature)

    # Calculate mean for each group while maintaining order
    center_features = [np.mean(group, axis=0) for group in grouped_features.values()]

    return center_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Center feature calculation.")
    parser.add_argument("--feature_path", "-feat", help="feature paths.", type=str)
    parser.add_argument("--dataset_name", "-dataset", help="name of the dataset.", type=str)
    parser.add_argument("--destination", "-dest", help="destination.", type=str, default="./")
    parser.add_argument("--num_cpus", "-n", help="number of CPUs to use", type=int, default=cpu_count())
    args = parser.parse_args()

    num_cpus = min(args.num_cpus, 10)  # Use specified number of CPUs, up to 10
    print(f"Using {num_cpus} CPUs")

    feature_paths = np.genfromtxt(args.feature_path, str) if not path.isdir(args.feature_path) else glob(
        f"{args.feature_path}/*.npy")

    output_file = f"{args.destination}/{args.dataset_name}_center_feature.npy"

    # Sort feature paths
    feature_paths = np.sort(feature_paths)

    # Process features
    center_features = process_features(feature_paths, num_cpus)

    print(f"Center features shape: {np.array(center_features).shape}")
    np.save(output_file, center_features)
