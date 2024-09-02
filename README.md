<div align="center">

# VEC2FACE: SCALING FACE DATASET GENERATION

[Haiyu Wu](https://haiyuwu.netlify.app/)<sup>1</sup> &emsp; [Jaskirat Singh](https://1jsingh.github.io/)<sup>2</sup> &emsp; [Sicong Tian](https://github.com/sicongT)<sup>3</sup>   

[Liang Zheng](https://zheng-lab.cecs.anu.edu.au/)<sup>2</sup> &emsp; [Kevin W. Bowyer](https://www3.nd.edu/~kwb/)<sup>1</sup> &emsp;  

<sup>1</sup>University of Notre Dame<br>
<sup>2</sup>The Australian National University<br>
<sup>3</sup>Indiana University South Bend

[//]: # (TODO)
<a href='https://haiyuwu.github.io/vec2face.github.io/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href=''><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/BooBooWu/Vec2Face'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange'></a>
<a href='https://huggingface.co/spaces/BooBooWu/Vec2Face'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>

</div>

This is the official implementation of **[Vec2Face](https://haiyuwu.github.io/vec2face.github.io/)**, an ID and attribute controllable face dataset generation model:

&emsp;✅ that generates face images purely based on the given image features<br>
&emsp;✅ that achieves the state-of-the-art performance in five standard test sets among synthetic datasets<br>
&emsp;✅ that first achieves higher accuracy than the same-scale real dataset (on CALFW)<br>
&emsp;✅ that can easily scale the dataset size to 10M images from 200k identities<br>


<img src='asset/architech.png'>

# News/Updates
- [2024/09/02] 🔥 We release Vec2Face [demo](https://huggingface.co/spaces/BooBooWu/Vec2Face)!
- [2024/09/01] 🔥 We release Vec2Face and HSFace datasets!

# :wrench: Installation
```bash
conda env create -f environment.yaml
```

# Download Model Weights
1) The weights of the Vec2Face model and estimators used in this work can be manually from [HuggingFace](https://huggingface.co/BooBooWu/Vec2Face) or using python:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/6DRepNet_300W_LP_AFLW2000.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/arcface-r100-glint360k.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/magface-r100-glint360k.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/vec2face_generator.pth", local_dir="./")
```
2) The weights of the FR models trained with HSFace (10k, 20k, 100k, 200k) can be downloaded using python:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="fr_weights/hsface10k.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="fr_weights/hsface20k.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="fr_weights/hsface100k.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="fr_weights/hsface200k.pth", local_dir="./")
```

# Download Datasets
1) The dataset used for **Vec2Face training** can be downloaded from manually from  [HuggingFace](https://huggingface.co/BooBooWu/Vec2Face) or using python:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="lmdb_dataset/WebFace4M/WebFace4M.lmdb", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="lmdb_dataset/WebFace4M/50000_ids_1022444_ims.npy", local_dir="./")
```
2) The **generated synthetic datasets** (HSFace10k and HSFace20k for now) can be downloaded using python:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="hsfaces/hsface10k.lmdb", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="hsfaces/hsface20k.lmdb", local_dir="./")
```

# ⚡Image Generation
Before generating images, the identity vectors need to be created/calculated and saved in a .npy file. We provide an example for you, but you can create your own center features.  
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="center_feature_examples.npy", local_dir="./")
```

Image generation with sampled identity features:
```commandline
python image_generation.py \
--model_weights weights/vec2face_generator.pth \
--batch_size 5 \
--example 1 \
--start_end 0:10 \
--name test \
--center_feature center_feature_examples.npy
```
Image generation with target yaw angle:
```commandline
python pose_image_generation.py \
--model_weights weights/vec2face_generator.pth \
--batch_size 5 \
--example 1 \
--start_end 0:10 \
--center_feature center_feature_examples.npy \
--name test \
--pose 45 \
--image_quality 25
```

# Training
## Vec2Face training
We only provide the WebFace4M dataset (see [here](https://github.com/HaiyuWu/vec2face?tab=readme-ov-file#download-datasets)) and the mask that we used for training the model, if you want to use other datasets, please referring the 
[prepare_training_set.py]('src=scripts/prepare_training_set.py') to convert the dataset to .lmdb.
Once the dataset is ready, modifying the following code to run the training:
```commandline
torchrun --nproc_per_node=1 --node_rank=0 --master_addr="host_addr" --master_port=3333 vec2face.py \
--rep_drop_prob 0.1 \
--use_rep \
--batch_size 8 \
--model vec2face_vit_base_patch16 \
--epochs 2000 \
--warmup_epochs 5 \
--blr 4e-5 \
--output_dir workspace/pixel_generator/24_try \
--train_source ./lmdb_dataset/WebFace4M/WebFace4M.lmdb \
--mask lmdb_dataset/WebFace4M/50000_ids_1022444_ims.npy \
--accum_iter 1
```

## FR model training
We borrowed the code from [SOTA-Face-Recognition-Train-and-Test](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test) to train the model. The random erasing function could be added after line 84 in [data_loader_train_lmdb.py](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test/blob/main/data/data_loader_train_lmdb.py), as shown below:
```python
transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.RandomErasing()
            ]
        )
```
Please follow the guidance of [SOTA-Face-Recognition-Train-and-Test](https://github.com/HaiyuWu/SOTA-Face-Recognition-Train-and-Test) for the rest of training process.

## TODO
- [ ] 100k and 200k datasets

# Acknowledgements
- Thanks to the WebFace4M creators for providing such a high-quality facial dataset❤️.
- Thanks to [Hugging Face](https://huggingface.co/) for providing a handy dataset and model weight management platform❤️.

# Citation
If you find Vec2Face useful for your research, please consider citing us and starring😄:

```bibtex
@misc{wu2024vec2face,
  title={Vec2Face: Scaling Face Dataset Generation with Loosely Constrained Vectors},
  author={Wu, Haiyu and Singh, Jaskirat and Tian, Sicong and Zheng, Liang and Bowyer, Kevin W.},
  year={2024}
}
```
