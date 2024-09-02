import sys
sys.path.append('./')
import gradio as gr
import random
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from models import iresnet
from sixdrepnet.model import SixDRepNet
import pixel_generator.vec2face.model_vec2face as model_vec2face
MAX_SEED = np.iinfo(np.int32).max
import torch


# change it to cuda for gpu
device = 'cuda'
def sample_nearby_vectors(base_vector, epsilons=[0.3, 0.5, 0.7], percentages=[0.4, 0.4, 0.2]):
    row, col = base_vector.shape
    norm = torch.norm(base_vector, 2, 1, True)
    diff = []
    for i, eps in enumerate(epsilons):
        diff.append(np.random.normal(0, eps, (int(row * percentages[i]), col)))
    diff = np.vstack(diff)
    np.random.shuffle(diff)
    diff = torch.tensor(diff)
    generated_samples = base_vector + diff
    generated_samples = generated_samples / torch.norm(generated_samples, 2, 1, True) * norm
    return generated_samples


def initialize_models():
    pose_model_weights = hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/6DRepNet_300W_LP_AFLW2000.pth", local_dir="./")
    id_model_weights = hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/arcface-r100-glint360k.pth", local_dir="./")
    quality_model_weights = hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/magface-r100-glint360k.pth", local_dir="./")
    generator_weights = hf_hub_download(repo_id="BooBooWu/Vec2Face", filename="weights/vec2face_generator.pth", local_dir="./")
    generator = model_vec2face.__dict__["vec2face_vit_base_patch16"](mask_ratio_mu=0.15, mask_ratio_std=0.25,
                                                mask_ratio_min=0.1, mask_ratio_max=0.5,
                                                use_rep=True,
                                                rep_dim=512,
                                                rep_drop_prob=0.,
                                                use_class_label=False)
    generator = generator.to(device)
    checkpoint = torch.load(generator_weights, map_location='cpu')
    generator.load_state_dict(checkpoint['model_vec2face'])
    generator.eval()

    id_model = iresnet("100", fp16=True).to(device)
    id_model.load_state_dict(torch.load(id_model_weights, map_location='cpu'))
    id_model.eval()

    quality_model = iresnet("100", fp16=True).to(device)
    quality_model.load_state_dict(torch.load(quality_model_weights, map_location='cpu'))
    quality_model.eval()

    pose_model = SixDRepNet(backbone_name='RepVGG-B1g2',
                            backbone_file='',
                            deploy=True,
                            pretrained=False
                            ).to(device)
    pose_model.load_state_dict(torch.load(pose_model_weights))
    pose_model.eval()

    return generator, id_model, pose_model, quality_model


def image_generation(input_image, quality, use_target_pose, pose, dimension):
    generator, id_model, pose_model, quality_model = initialize_models()

    generated_images = []
    if input_image is None:
        feature = np.random.normal(0, 1.0, (1, 512))
    else:
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = torch.from_numpy(input_image).unsqueeze(0).float().to(device)
        input_image.div_(255).sub_(0.5).div_(0.5)
        feature = id_model(input_image).clone().detach().cpu().numpy()

    if not use_target_pose:
        features = []
        norm = np.linalg.norm(feature, 2, 1, True)
        for i in np.arange(0, 4.8, 0.8):
            updated_feature = feature
            updated_feature[0][dimension] = feature[0][dimension] + i

            updated_feature = updated_feature / np.linalg.norm(updated_feature, 2, 1, True) * norm

            features.append(updated_feature)
        features = torch.tensor(np.vstack(features)).float().to(device)
        if quality > 25:
            images, _ = generator.gen_image(features, quality_model, id_model, q_target=quality)
        else:
            _, _, images, *_ = generator(features)
    else:
        features = torch.repeat_interleave(torch.tensor(feature), 6, dim=0)
        features = sample_nearby_vectors(features, [0.7], [1]).float().to(device)
        if quality > 25 or pose > 20:
            images, _ = generator.gen_image(features, quality_model, id_model, pose_model=pose_model,
                                            q_target=quality, pose=pose, class_rep=features)
        else:
            _, _, images, *_ = generator(features)

    images = ((images.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
    for image in images:
        generated_images.append(Image.fromarray(image))
    return generated_images


def process_input(image_input, num1, num2, num3, num4, random_seed, target_quality, use_target_pose, target_pose):
    # Ensure all dimension numbers are within [0, 512)
    num1, num2, num3, num4 = [max(0, min(int(n), 511)) for n in [num1, num2, num3, num4]]

    # Use the provided random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    if image_input is None:
        input_data = None
    else:
        # Process the uploaded image
        input_data = Image.open(image_input)
        input_data = np.array(input_data.resize((112, 112)))

    generated_images = image_generation(input_data, target_quality, use_target_pose, target_pose, [num1, num2, num3, num4])

    return generated_images

def select_image(value, images):
    # Convert the float value (0 to 4) to an integer index (0 to 9)
    index = int(value / 0.8)
    return images[index]

def toggle_inputs(use_pose):
    return [
        gr.update(visible=use_pose, interactive=use_pose),  # target_pose
        gr.update(interactive=not use_pose),  # num1
        gr.update(interactive=not use_pose),  # num2
        gr.update(interactive=not use_pose),  # num3
        gr.update(interactive=not use_pose),  # num4
    ]


def main():
    with gr.Blocks() as demo:
        title = r"""
            <h1 align="center">Vec2Face: Scaling Face Dataset Generation with Loosely Constrained Vectors</h1>
            """

        description = r"""
            <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/HaiyuWu/vec2face' target='_blank'><b>Vec2Face: Scaling Face Dataset Generation with Loosely Constrained Vectors</b></a>.<br>

            How to use:<br>
            1. Upload an image with a cropped face image or directly click <b>Submit</b> button, six images will be shown on the right. 
            2. You can control the image quality, image pose, and modify the values in the target dimensions to change the output images. 
            3. The output results will shown six results of dimension modification or pose images.
            4. Since the demo is CPU-based, higher quality and larger pose need longer time to run.
            5. Enjoy! 😊
            """

        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                image_file = gr.Image(label="Upload an image (optional)", type="filepath")

                gr.Markdown("""
                ## Dimension Modification
                Enter the values for the dimensions you want to modify (0-511). 
                """)

                with gr.Row():
                    num1 = gr.Number(label="Dimension 1", value=0, minimum=0, maximum=511, step=1)
                    num2 = gr.Number(label="Dimension 2", value=0, minimum=0, maximum=511, step=1)
                    num3 = gr.Number(label="Dimension 3", value=0, minimum=0, maximum=511, step=1)
                    num4 = gr.Number(label="Dimension 4", value=0, minimum=0, maximum=511, step=1)

                random_seed = gr.Number(label="Random Seed", value=42, minimum=0, maximum=MAX_SEED, step=1)
                target_quality = gr.Slider(label="Minimum Quality", minimum=22, maximum=35, step=1, value=24)

                with gr.Row():
                    use_target_pose = gr.Checkbox(label="Use Target Pose")
                    target_pose = gr.Slider(label="Target Pose", value=0, minimum=0, maximum=90, step=1, visible=False)

                submit = gr.Button("Submit", variant="primary")

                gr.Markdown("""
                            ## Usage tips of Vec2Face
                            - Directly clicking "Submit" button will give you results from a randomly sampled vector. 
                            - If you want to modify more dimensions, please write your own code. Code snippets in [Vec2Face repo](https://github.com/HaiyuWu/vec2face) might be helpful.
                            - If you want to create extreme pose image (e.g., >70), please do not set image quality larger than 27.
                            - <span style="color: red;">!</span> <span style="color: red;">!</span> <span style="color: red;">!</span> **Due to the limitation of SixDRepNet (pose estimator), pose editing results might be corrupted/incorrect. For better performance, you can integrade other pose estimators.** <span style="color: red;">!</span> <span style="color: red;">!</span> <span style="color: red;">!</span>
                            - For better experience, we suggest you to run code on a GPU machine.
                            """)

            with gr.Column():
                gallery = gr.Image(label="Generated Image")
                incremental_value_slider = gr.Slider(
                    label="Result of dimension modification or results of pose images",
                    minimum=0, maximum=4, step=0.8, value=0
                )
                gr.Markdown("""
                            - These values are added to the dimensions (before normalization), **please ignore it if pose editing is on**.
                            """)

        use_target_pose.change(
            fn=toggle_inputs,
            inputs=[use_target_pose],
            outputs=[target_pose, num1, num2, num3, num4]
        )

        generated_images = gr.State([])

        submit.click(
            fn=process_input,
            inputs=[image_file, num1, num2, num3, num4, random_seed, target_quality, use_target_pose, target_pose],
            outputs=[generated_images]
        ).then(
            fn=select_image,
            inputs=[incremental_value_slider, generated_images],
            outputs=[gallery]
        )

        incremental_value_slider.change(
            fn=select_image,
            inputs=[incremental_value_slider, generated_images],
            outputs=[gallery]
        )
        article = r"""
        ---
        📝 **Citation**
        <br>
        If our work is helpful for your research or applications, please cite us via:
        ```bibtex
        @article{wu2024vec2face,
        title={Vec2Face: Scaling Face Dataset Generation with Loosely Constrained Vectors},
        author={Wu, Haiyu and Singh, Jaskirat and Tian, Sicong and Zheng, Liang and Bowyer, Kevin W.},
        year={2024}
        }
        ```
        📧 **Contact**
        <br>
        If you have any questions, please feel free to open an issue or directly reach us out at <b>hwu6@nd.edu</b>.
        """
        gr.Markdown(article)

    demo.launch(share=True)


if __name__ == "__main__":
    main()