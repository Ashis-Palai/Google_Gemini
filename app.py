import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import numpy as np

# Fetching the API key from the environment variable
api_key = os.environ.get('GOOGLE_API_KEY')

if api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it before running the script.")

# Configuring the GenerativeAI API with the obtained API key
genai.configure(api_key=api_key)

# Creating a GenerativeModel instance
model = genai.GenerativeModel('gemini-pro-vision')

# Function to generate content based on input image and optional prompt
def generate_content(image, prompt=None):
    # Convert Streamlit Image to PIL Image
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = Image.open(image)

    if prompt:
        # Generating content using the model with prompt
        response = model.generate_content([prompt, image_pil])
        response.resolve()

        # Extracting the generated text
        generated_text = response.text.strip()
    else:
        # If no prompt is provided, generate content without prompt
        response = model.generate_content([image_pil])
        response.resolve()

        # Extracting the generated text
        generated_text = response.text.strip()

    return generated_text

# Streamlit Interface with enhanced styling
st.set_page_config(
    page_title="Generative AI Image Generation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with app information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses Generative AI to generate text based on an uploaded image and an optional prompt."
)

# Upload image from the user
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Display the uploaded image with reduced size
if uploaded_image is not None:
    # Resize the image for display
    img = Image.open(uploaded_image)
    resized_img = img.resize((400, 400))

    st.image(resized_img, caption="Uploaded Image", use_column_width=True)

    # Allow the user to enter a prompt
    prompt = st.text_input("Enter a prompt (optional)", "")

    # Submit button to trigger content generation
    if st.button("Generate Content", key="generate_button"):
        # Add a loading spinner while processing
        with st.spinner("Generating content..."):
            # Process the image and generate content
            generated_text = generate_content(uploaded_image, prompt)

        # Display the generated text with a success message
        st.success("Content generated successfully!")
        st.text("Generated Text:")
        st.write(generated_text)
