import os
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_unrestricted_images(prompt):
    # Load pre-trained model
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    
    # Generate 2 images with no restrictions
    images = []
    for i in range(2):
        image = pipe(prompt, height=768, width=768, num_inference_steps=50).images[0]
        images.append(image)
    
    return images

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    images = generate_unrestricted_images(prompt)
    
    # Save images
    for i, image in enumerate(images):
        image.save(f"generated_image_{i}.png")
        print(f"Generated image {i+1}: generated_image_{i}.png")
