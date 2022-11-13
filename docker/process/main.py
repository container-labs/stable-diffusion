from diffusers import StableDiffusionPipeline
from GPUtil import showUtilization as gpu_usage
from argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
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
        type=str,
        default="100",
        help="",
    )
    parser.add_argument(
        "--num_images",
        type=str,
        default="50",
        help="",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="boredape",
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

gpu_usage()

pipe = StableDiffusionPipeline.from_pretrained(
  #"runwayml/stable-diffusion-v1-5",
  "./gcloud-vol/job-1668365875",
  use_auth_token=""
).to("mps")

for i in range(args.num_images):
  # run it
  result = pipe(
        f"{args.style}, {args.phrase}",
        num_inference_steps=args.max_steps,
        )
  image = result.images[0]

  gpu_usage()
  # notebook only
  # display(image)
  image.save(f"{args.output_dir}/ape-{i}.png")

# this paired down example will be the first prod pipeline
# after a fucking week of exploration and building and docker hell
# I can run another job as save it to gcs
