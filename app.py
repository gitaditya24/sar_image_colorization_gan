import torch
import torch.nn as nn
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
from model import UnetGenerator  # Import your generator model

# Load model checkpoint
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("checkpoint_epoch_200.pth", map_location=device)
    
    model = UnetGenerator(c_in=1, c_out=3)  # Ensure correct input/output channels
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()
    return model.to(device)

# Preprocess image
def preprocess_image(image, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Generate Colorized Image
def generate_colorized_image(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()  # Load model
    input_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)  # Pass through generator
    
    output_tensor = output_tensor.squeeze(0).cpu().detach()
    output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)  # De-normalize
    return transforms.ToPILImage()(output_tensor)

# Streamlit UI
st.title("SAR Image Colorization App")
st.write("Upload a SAR grayscale image, and the model will generate a colorized version.")

uploaded_file = st.file_uploader("Choose a SAR image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    sar_image = Image.open(uploaded_file).convert("L")  # Ensure grayscale
    
    # Generate colorized output
    colorized_image = generate_colorized_image(sar_image)

    # Display images side by side
    col1, col2 = st.columns(2)  # Create two columns
    with col1:
        st.image(sar_image, caption="Uploaded SAR Image", use_container_width=True)
    with col2:
        st.image(colorized_image, caption="Colorized Image", use_container_width=True)

    # Download button
    colorized_image.save("colorized_output.png")  # Save the image first
    with open("colorized_output.png", "rb") as file:
        st.download_button(
            label="Download Colorized Image",
            data=file,
            file_name="colorized_output.png",
            mime="image/png"
        )


