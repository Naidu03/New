import base64
import os
import tempfile
import warnings
import io
from PIL import Image
import streamlit as st
from google import genai
from google.genai import types
from google.genai.types import HarmBlockThreshold

# -------------------- Streamlit UI Customization --------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #1F77B4;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF7F0E;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    .stTextInput>div>div>input {
        background-color: #1E1E1E;
        color: #FAFAFA;
    }
    .stMarkdown {
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Core Function --------------------
def swap_clothing(person_image, clothing_image, chest, waist, hips, height, output_mode):
    """
    Generate an image where the person is wearing clothing,
    scaled to match given measurements. Supports 2D & 3D.
    """
    warning_buffer = io.StringIO()
    warnings.filterwarnings('always')  
    temp_files = []
    uploaded_files = []
    client = None
    output_image = None
    output_text = ""

    with warnings.catch_warnings(record=True) as warning_list:
        try:
            if person_image is None or clothing_image is None:
                return None, "Please upload both images."

            api_key = ''  # <-- replace with your GEMINI_API_KEY or set via environment
            if not api_key:
                return None, "GEMINI_API_KEY not found. Please set it in your environment."

            client = genai.Client(api_key=api_key)

            # Save images to temporary files
            for img, prefix in [(person_image, "person"), (clothing_image, "clothing")]:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    img.save(temp_file.name)
                    temp_files.append(temp_file.name)

            uploaded_files = [
                client.files.upload(file=temp_files[0]),  # person
                client.files.upload(file=temp_files[1]),  # clothing
            ]

            # ---- PROMPT WITH MEASUREMENTS ----
            prompt = f"""
            Generate a photorealistic try-on result.

            PERSON:
            - Preserve identity (face, hair, skin tone, expression).
            - Keep background, lighting, and pose unchanged.

            CLOTHING FIT (match measurements below):
            - Chest: {chest} inches
            - Waist: {waist} inches
            - Hips: {hips} inches
            - Height: {height} inches

            Ensure clothing scales naturally to fit these proportions.
            Retain fabric folds, wrinkles, and texture.

            REALISM REQUIREMENTS:
            - No distortion, misaligned patterns, or artifacts.
            - Match lighting & shadows from original photo.
            - Colors should look natural in the scene.

            OUTPUT MODE: {output_mode}
            """
            if output_mode == "3D Model":
                prompt += """
                Render as a 3D-style model or mannequin, showing body depth and garment draping
                in three dimensions. Maintain realism while giving a 3D perspective.
                """

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text("This is the person image. Do not change face or background."),
                        types.Part.from_uri(uploaded_files[0].uri, uploaded_files[0].mime_type),
                        types.Part.from_text("This is the clothing image. Apply it on the person."),
                        types.Part.from_uri(uploaded_files[1].uri, uploaded_files[1].mime_type),
                        types.Part.from_text(prompt),
                    ],
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                temperature=0.099,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_modalities=["image", "text"],
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=HarmBlockThreshold.BLOCK_NONE),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=HarmBlockThreshold.BLOCK_NONE),
                ],
                response_mime_type="text/plain",
            )

            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=contents,
                config=generate_content_config,
            )

            if response and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if part.text is not None:
                            output_text += part.text + "\n"
                        elif part.inline_data is not None:
                            try:
                                image_data = part.inline_data.data
                                if not isinstance(image_data, bytes):
                                    image_data = base64.b64decode(image_data)
                                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                                    temp_file.write(image_data)
                                    temp_file_path = temp_file.name
                                with Image.open(temp_file_path) as img:
                                    output_image = img.copy()
                                os.unlink(temp_file_path)
                            except Exception as img_error:
                                output_text += f"Error processing image: {str(img_error)}\n"
            else:
                output_text = "The model did not generate a valid response. Try again."

        except Exception as e:
            error_details = f"Error: {str(e)} | Type: {type(e).__name__}"
            if warning_list:
                for warning in warning_list:
                    error_details += f"\n- Warning: {warning.message}"
            return None, error_details

        finally:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            for uploaded_file in uploaded_files:
                try:
                    if hasattr(client.files, 'delete') and uploaded_file:
                        client.files.delete(uploaded_file.uri)
                except:
                    pass
            client = None

        return output_image, output_text

# -------------------- Streamlit UI --------------------
st.title("👗 Cloth Try-On AI")
st.image('blob.png', width=200)
st.markdown("Upload your photo & clothing, enter your measurements, and try them on virtually!")

col1, col2 = st.columns(2)
with col1:
    person_file = st.file_uploader("Upload Your Photo", type=["jpg", "jpeg", "png"])
    if person_file:
        st.image(person_file, caption="Your Photo", width=300)
with col2:
    clothing_file = st.file_uploader("Upload Clothing Photo", type=["jpg", "jpeg", "png"])
    if clothing_file:
        st.image(clothing_file, caption="Clothing Photo", width=300)

# Measurement inputs
st.subheader("📏 Enter Your Measurements")
chest = st.number_input("Chest (inches)", min_value=20, max_value=60, value=38)
waist = st.number_input("Waist (inches)", min_value=20, max_value=60, value=32)
hips = st.number_input("Hips (inches)", min_value=20, max_value=60, value=40)
height = st.number_input("Height (inches)", min_value=50, max_value=84, value=68)

# Output Mode
output_mode = st.radio("Choose Output Mode", ["2D Image", "3D Model"])

# Generate Button
if st.button("Generate"):
    if person_file and clothing_file:
        person_image = Image.open(person_file)
        clothing_image = Image.open(clothing_file)
        output_image, output_text = swap_clothing(person_image, clothing_image, chest, waist, hips, height, output_mode)

        if output_image:
            st.image(output_image, caption="Generated Output", width=500)
        st.text(output_text)
    else:
        st.error("Please upload both images.")
