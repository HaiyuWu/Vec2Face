from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/6DRepNet_300W_LP_AFLW2000.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/arcface-r100-glint360k.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/magface-r100-glint360k.pth", local_dir="./")
hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/vec2face_generator.pth", local_dir="./")