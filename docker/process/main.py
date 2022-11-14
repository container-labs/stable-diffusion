import argparse
import os

from diffusers import StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_out",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_steps",
         type=int,
        default="100",
        help="",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default="50",
        help="",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="epaderod",
        help="",
    )
    parser.add_argument(
        "--phrase",
        type=str,
        default="Elon Musk",
        help="",
    )

    args = parser.parse_args()
    return args


args = parse_args()
pipe = StableDiffusionPipeline.from_pretrained(
  args.pretrained_model_name_or_path,
  use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
).to("cuda")

for i in range(args.num_images):
  image_path = f"{args.output_dir}/{args.style}-{i}.png"
  result = pipe(
        f"{args.style} {args.phrase}",
        num_inference_steps=args.max_steps,
        )
  image = result.images[0]
  image.save(f"{args.output_dir}/{args.style}-{i}.png")

