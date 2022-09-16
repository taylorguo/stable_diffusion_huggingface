# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline


pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4")
pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
prompt = "a bear is playing guitar, oil on canvas"
with autocast("cuda"):
    image = pipe(prompt).images[0]

# image.save("astronaut_rides_horse.png")
image.save("bear_playing_guitar.png")