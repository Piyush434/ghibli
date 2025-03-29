import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline

# Load Stable Diffusion Model
@st.cache_resource()
def load_model():
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

pipe = load_model()

# Streamlit UI
st.title("🎨 Ghibli-Style Image Generator")
st.title("Made with ❤️ by Piyush...")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process Image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Generate Ghibli-style image
    prompt = "A Ghibli-style anime painting, soft lighting, vibrant colors, highly detailed, magical world, fantasy"
    with st.spinner("Generating Ghibli-style image..."):
        ghibli_image = pipe(prompt=prompt, image=image_tensor, strength=0.75).images[0]

    # Convert to OpenCV format for post-processing
    ghibli_cv = np.array(ghibli_image)
    ghibli_cv = cv2.cvtColor(ghibli_cv, cv2.COLOR_RGB2BGR)
    ghibli_cv = cv2.edgePreservingFilter(ghibli_cv, flags=1, sigma_s=60, sigma_r=0.4)

    # Save and display result
    output_path = "ghibli_style.jpg"
    cv2.imwrite(output_path, ghibli_cv)
    st.image(ghibli_image, caption="Ghibli-Style Output", use_column_width=True)
    st.success("✅ Image Generation Complete!")

    # Download Button
    with open(output_path, "rb") as file:
        btn = st.download_button(
            label="📥 Download Ghibli Image",
            data=file,
            file_name="ghibli_style.jpg",
            mime="image/jpeg"
        )
