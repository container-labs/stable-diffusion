from diffusers import StableDiffusionPipeline
from GPUtil import showUtilization as gpu_usage

device = "cuda"

print("Initial GPU Usage")
gpu_usage()

pipe = StableDiffusionPipeline.from_pretrained(
  "runwayml/stable-diffusion-v1-5",
  use_auth_token="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"
).to(device)

# run it
result = pipe(
        "lvngvncnt, Elon Musk",
        num_inference_steps=100)
image = result.images[0]

gpu_usage()
# notebook only
# display(image)


# !python textual_inversion.py \
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#   --train_data_dir=./src/data \
#   --learnable_property="object" \
#   --placeholder_token="<bored-ape>" --initializer_token="ape" \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --max_train_steps=1000 \
#   --learning_rate=5.0e-04 --scale_lr \
#   --output_dir="textual_inversion_bored_ape"
