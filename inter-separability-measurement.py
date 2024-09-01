import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def get_similarity(center_features):
    sim = cosine_similarity(center_features, center_features)
    res = sim > 0.40
    unique_samples = np.triu(res, k=1).sum(axis=0) == 0
    return sum(unique_samples)


def draw_similarities(center_features, dataset_names):
    # loading
    plt.figure(figsize=(8, 6))
    labels = []
    for center_feature_path, dataset_name in zip(center_features, dataset_names):
        print(f"Dealing with {dataset_name}...")
        labels.append(f"{dataset_name}")
        center_feature = np.load(center_feature_path)
        for i in range(0, 200000, 1000):
            center_feature_chunk = center_feature[0:i + 1000]
            info = get_similarity(center_feature_chunk)
            print(f"{info} unique identities from {i + 1000} IDs.")


if __name__ == '__main__':
    draw_similarities([
        "../Arc2Face/arc2face200k/Arc2Face.npy",
        "./id-coine-sim-calc/Vec2Face.npy",
        "./lmdb_dataset/WebFace4M/center_features.npy",
    ],
        [
            "Arc2Face",
            "Vec2Face",
            "Real"
        ])
