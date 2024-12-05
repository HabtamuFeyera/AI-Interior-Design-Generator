import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_image(prompt, output_path="generated_image.png"):
    # Load pre-trained model 
    model_id = "CompVis/stable-diffusion-v1-4"  
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to("cuda")  
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Save the generated image
    image.save(output_path)
    print(f"Image generated and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interior design images using AI.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt describing the interior design.")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Path to save the generated image.")
    
    args = parser.parse_args()
    generate_image(args.prompt, args.output)
