
from runner import Runner

opts = {
    "model_name" : "nitrosocke/redshift-diffusion",
   "prompt" : "redshift style, Elon Musk as Batman",
    "steps": 10,
    "number_of_images": 40,
    "seed_start": 0,
    "seed_end": 2048,
}

runner = Runner(opts)
runner.setup_pipeline()
runner.run()

# as the Joker from Batman
# as Batman

# pipe = pipe.to("mps")
# for x in range(0, 2048, 16):
#     generator = Generator().manual_seed(x)
#     _ = pipe(prompt, num_inference_steps=1,guidance_scale=7.5, generator=generator)
#     result = pipe(prompt, num_inference_steps=10, generator=generator)
#     image = result.images[0]
#     # python "f strings"...
#     image.save(f"styles-redshift-mps-{x}.png")


# for x in range(0, 2048, 16):
#     generator = Generator().manual_seed(x)
#     # _ = pipe(prompt, num_inference_steps=1, generator=generator, device="mps")
#     result = pipe(prompt, num_inference_steps=10, generator=generator)
#     image = result.images[0]
#     # python "f strings"...
#     image.save(f"styles-redshift-{x}.png")


# good_ones = [48, 1392, 1328]
# for x in good_ones:
#     generator = Generator().manual_seed(x)
#     _ = pipe(prompt, num_inference_steps=1, generator=generator)
#     result = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator)
#     image = result.images[0]
#     # python "f strings"...
#     image.save(f"styles-redshift-hq-{x}.png")



