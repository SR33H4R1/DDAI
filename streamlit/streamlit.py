import streamlit as st
from PIL import Image, ImageDraw
import io
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_login_auth_ui.widgets import __login__
import json
from streamlit_option_menu import option_menu

# --- Set Page Config ---
st.set_page_config(
    page_title="IRIS - AI-Powered Medical Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=":hospital:",
)

# --- Authentication ---
__login__obj = __login__(
    auth_token="courier_auth_token",
    company_name="Shims",
    width=200,
    height=250,
    hide_menu_bool=False,
    hide_footer_bool=False,
    lottie_url='https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json'
)
LOGGED_IN = __login__obj.build_login_ui()

# --- Load user data ---
with open('/Users/sreehari/PycharmProjects/FInal_Project/_secret_auth_.json', 'r') as f:
    user_data = json.load(f)

# --- Load Models  ---
if LOGGED_IN:
    chest_model = load_model('/Users/sreehari/PycharmProjects/FInal_Project/models/chest.h5')
    alz_model = load_model('/Users/sreehari/PycharmProjects/FInal_Project/models/alzheimer.h5')
    arth_model = load_model('/Users/sreehari/PycharmProjects/FInal_Project/models/arthritis.h5')


    # --- Preprocessing Function ---
    def preprocess(image_bytes, target_size=(128, 128)):
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array / 255.0
        return np.expand_dims(img_array, axis=0)


    # --- Prediction Functions ---
    def predict_chest(image_bytes):
        preprocessed_image = preprocess(image_bytes)
        predictions = chest_model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions[0])
        class_names = ['COVID-19', 'Pneumonia', 'Normal']
        return class_names[predicted_class]


    def predict_alzheimers(image_bytes):
        preprocessed_image = preprocess(image_bytes)
        predictions = alz_model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions[0])
        class_names = ['Mild', 'Moderate', 'NonDementia']
        return class_names[predicted_class]


    def predict_arthritis(image_bytes):
        preprocessed_image = preprocess(image_bytes, target_size=(150, 150))
        predictions = arth_model.predict(preprocessed_image)
        if predictions[0][0] > 0.5:
            predicted_class = 'Arthritis'
        else:
            predicted_class = 'Normal'
        return predicted_class


    # --- Streamlit App ---
    # --- Sidebar Navigation ---
    with st.sidebar:
        # --- User Info ---
        st.write("---")

        # Center the image and username
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            user_image = Image.open('/Users/sreehari/PycharmProjects/FInal_Project/streamlit/images/profile.png')
            user_image = user_image.resize((512, 512))
            user_image = user_image.convert("RGBA")

            # Create a circular mask
            mask = Image.new("L", user_image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0) + user_image.size, fill=255)

            # Apply the mask to the image
            user_image.putalpha(mask)
            st.image(user_image, use_column_width=True, output_format='PNG')
            st.markdown(f"<h5 style='text-align: center;'>{user_data[0]['name']}</h5>", unsafe_allow_html=True)

        st.write("---")  # Add another separator
        selected_page = option_menu(
            menu_title="Navigation",
            options=[
                "Home",
                "Chest Disease Prediction",
                "Alzheimer's Prediction",
                "Arthritis Prediction",
                "About Us"
            ],
            icons=['house', 'lungs', 'activity', 'person-walking', 'info-circle'],
            menu_icon="cast",
            default_index=0,
        )
    # --- Page Content ---
    if selected_page == "Chest Disease Prediction":
        st.title("Chest Disease Prediction")
        st.write("This model can classify chest X-ray images into categories such as COVID-19, Pneumonia, and Normal.")
        image_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            image_bytes = image_file.read()
            st.image(image_bytes, caption="Uploaded Chest X-ray", use_column_width=True)
            prediction = predict_chest(image_bytes)
            st.success(f"**Predicted Class: {prediction}**")


    elif selected_page == "Alzheimer's Prediction":
        st.title("Alzheimer's Prediction")
        st.write(
            "This model analyzes brain scan images to predict the severity of Alzheimer's disease (Mild, Moderate, NonDementia).")
        image_file = st.file_uploader("Upload Brain Scan Image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            image_bytes = image_file.read()
            st.image(image_bytes, caption="Uploaded Brain Scan", use_column_width=True)
            prediction = predict_alzheimers(image_bytes)
            st.success(f"**Predicted Class: {prediction}**")


    elif selected_page == "Arthritis Prediction":
        st.title("Arthritis Prediction")
        st.write("This model analyzes X-ray images to predict the presence of Arthritis.")
        image_file = st.file_uploader("Upload Joint Image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            image_bytes = image_file.read()
            st.image(image_bytes, caption="Uploaded Joint Image", use_column_width=True)
            prediction = predict_arthritis(image_bytes)
            st.success(f"**Predicted Class: {prediction}**")


    elif selected_page == "About Us":
        st.title("About Us")
        st.write("Meet the team behind IRIS:")

        # Team members with names and image paths
        team_members = [
            {
                "name": "Sreehari A L",
                "image_path": "/Users/sreehari/PycharmProjects/FInal_Project/streamlit/images/Sreehari.jpeg",

            },
            {
                "name": "Gautham S B",
                "image_path": "/Users/sreehari/PycharmProjects/FInal_Project/streamlit/images/gautham.jpeg",

            },
            {
                "name": "Adithyan M",
                "image_path": "/Users/sreehari/PycharmProjects/FInal_Project/streamlit/images/Adithyan_M.jpeg",

            },
            {
                "name": "Adithyan S",
                "image_path": "/Users/sreehari/PycharmProjects/FInal_Project/streamlit/images/Adithyan_S.jpeg",

            },
        ]

        # Create two columns for team members
        col1, col2 = st.columns(2)

        # Display team members in two columns
        for i, member in enumerate(team_members):
            with col1 if i % 2 == 0 else col2:
                # Create a container for the image and name
                with st.container():
                    image = Image.open(member["image_path"])
                    image = image.resize((1024, 1024))
                    image = image.convert("RGBA")

                    # Create a circular mask
                    mask = Image.new("L", image.size, 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((0, 0) + image.size, fill=255)
                    image.putalpha(mask)

                    st.image(image, width=300, output_format='PNG')

                    st.markdown(
                        f"<div style='text-align:left;padding-left:110px;padding-bottom:50px'>{member['name']}</div>",
                        unsafe_allow_html=True)


    else:  # selected_page == "Home"
        st.title("IRIS: AI-Powered Medical Image Analysis")
        st.write("## Empowering Healthcare with Deep Learning")
        st.image("/Users/sreehari/PycharmProjects/FInal_Project/streamlit/images/home.jpg",
                 use_column_width=True,
                 caption="Leveraging AI for faster and more accurate diagnoses.")

        st.write("""
        Welcome to IRIS, a platform designed to assist healthcare professionals in making
        more informed decisions using the power of artificial intelligence.
        **Our mission:** To provide accessible, accurate, and efficient disease
        prediction tools based on medical images.
        """)
