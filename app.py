import pandas as pd
import random
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from wordcloud import WordCloud
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import base64
import plotly.express as px
import plotly.graph_objects as go

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Set page configuration
st.set_page_config(page_title="APP HACKATHON 2025", layout="wide")

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    page_bg_img = f"""
    <style>
    .stApp, .stSidebar {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("back.jpg")  # Provide the correct path to your image

st.markdown(
    """
    <style>
    .big-font { font-size:40px !important; font-weight: bold; text-align: center; }
    .medium-font { font-size:30px !important; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Custom Background Video Styling
video_path = "background.mp4"  # Ensure this file is in the working directory
if os.path.exists(video_path):
    st.video(video_path)

# Sidebar
st.sidebar.title("APP HACKATHON 2025 Dashboard")
st.sidebar.write(
    """
    - Welcome to the Hackathon Analysis Dashboard!
    - Gain deep insights into participant trends and feedback analysis.
    - Visualize data across various domains, states, and colleges.
    - Generate word clouds to analyze participant sentiments.
    - Upload and process images related to hackathon events.
    - Fully interactive and user-friendly with dark-themed aesthetics.
    - Supports 400 participants over 3 days in 5 hackathon domains.
    """
)

# Dataset Generation
def generate_dataset():
    domains = ['AI/ML', 'Web Development', 'Cybersecurity', 'IoT', 'Blockchain']
    states = ['Maharashtra', 'Karnataka', 'Haryana', 'Delhi', 'TamilNadu']
    colleges = ['NMIM', 'Christ University', 'CU', 'DU', 'VIT']
    days = ['Day 1', 'Day 2', 'Day 3']
    feedbacks = ["Amazing experience!", "Had a great time!", "Very well organized!", "Loved the sessions!", "Could be better", "Awesome mentors!", "Good learning opportunity", "Enjoyed networking!", "Informative and engaging", "A bit chaotic", "Fantastic event!", "Highly recommended!", "Need more hands-on sessions", "Great exposure", "Loved the energy!", "Workshops were insightful", "The pace was too fast", "Very interactive", "Support was excellent", "Well-structured hackathon"]

    data = []
    for i in range(350):
        participant = {
            'Participant_ID': i + 1,
            'Name': f'Participant_{i + 1}',
            'Domain': random.choice(domains),
            'Day': random.choice(days),
            'College': random.choice(colleges),
            'State': random.choice(states),
            'Age': random.randint(18, 30),
            'Experience (Years)': round(random.uniform(0, 5), 1),
            'Rating (Out of 5)': round(random.uniform(1, 5), 1),
            'Feedback': random.choice(feedbacks)
        }
        data.append(participant)

    df = pd.DataFrame(data)
    df.to_csv('hackathon_participants.csv', index=False)
    st.sidebar.success("Dataset generated successfully!")

if st.sidebar.button("Generate Dataset"):
    generate_dataset()


st.markdown('<p class="big-font">Hackathon Participants Analysis 🪙</p>', unsafe_allow_html=True)
st.subheader("Explore the dataset and visualizations")
st.write("This dashboard provides insights into the participants of the hackathon, including their domains, states, colleges, and feedback.")
# Load dataset
try:
    df = pd.read_csv('hackathon_participants.csv')
    st.subheader("Hackathon Participants Dataset")
    st.dataframe(df)
except FileNotFoundError:
    st.error("Dataset not found. Please generate the dataset first.")

# Dataset Visualization
st.sidebar.header("Dataset Visualization")
visualization_options = st.sidebar.multiselect("Choose visualizations", ["Domain Distribution", "Age Distribution", "Experience vs Rating", "Word Cloud (Feedback)"])

if visualization_options:
    st.subheader("Dataset Visualizations")

    for option in visualization_options:
        if option == "Domain Distribution":
            fig = px.bar(df, x="Domain", title="Domain Distribution", color="Domain",
                         template="plotly_white")
        
        elif option == "Age Distribution":
            fig = px.histogram(df, x="Age", nbins=10, title="Age Distribution", 
                               marginal="rug", color_discrete_sequence=["blue"], 
                               template="plotly_white")

        elif option == "Experience vs Rating":
            fig = px.scatter(df, x="Experience (Years)", y="Rating (Out of 5)", 
                             color="Domain", title="Experience vs Rating", 
                             size="Rating (Out of 5)", hover_data=['College'], 
                             template="plotly_white")

        elif option == "Word Cloud (Feedback)":
            from wordcloud import WordCloud
            text = " ".join(df["Feedback"].astype(str))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig = go.Figure(go.Image(z=wordcloud.to_array()))  # Convert to Plotly Image
            fig.update_layout(template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)  


# Text Analysis
st.sidebar.header("Text Analysis")
text_analysis_options = st.sidebar.multiselect("Choose text analysis methods", ["Word Frequency", "Stemming", "Lemmatization", "Named Entity Recognition"])

if text_analysis_options:
    st.subheader("Text Analysis")
    feedback_text = " ".join(df["Feedback"].astype(str))
    tokens = word_tokenize(feedback_text)
    
    for option in text_analysis_options:
        if option == "Word Frequency":
            word_freq = pd.Series(tokens).value_counts().head(10)
            st.write("Top 10 Frequent Words in Feedback:")
            st.bar_chart(word_freq)
        elif option == "Stemming":
            stemmer = PorterStemmer()
            stemmed_words = [stemmer.stem(word) for word in tokens]
            st.write("Stemming Output:")
            st.write(" ".join(stemmed_words[:50]))
        elif option == "Lemmatization":
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
            st.write("Lemmatization Output:")
            st.write(" ".join(lemmatized_words[:50]))
        elif option == "Named Entity Recognition":
            ner_tree = ne_chunk(pos_tag(tokens))
            st.write("Named Entity Recognition (NER):")
            st.write(ner_tree)

# Image Processing Section
st.sidebar.header("Image Processing")
selected_option = st.sidebar.radio("Select an option", ["Gallery", "Upload New Image"])

# Image Gallery
st.subheader("Image Gallery")
uploaded_images = st.file_uploader("Upload Images for Gallery", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
gallery_images = []
if uploaded_images:
    cols = st.columns(3)
    for idx, uploaded_image in enumerate(uploaded_images):
        with cols[idx % 3]:
            st.image(uploaded_image, caption=f"Uploaded: {uploaded_image.name}", use_container_width=True)
            gallery_images.append(uploaded_image)

# Image Processing
st.subheader("Image Processing")
image_to_process = None
if selected_option == "Gallery" and gallery_images:
    selected_img = st.selectbox("Select an Image from Gallery", gallery_images)
    image_to_process = Image.open(selected_img)
elif selected_option == "Upload New Image":
    new_image = st.file_uploader("Upload an Image for Processing", type=["jpg", "jpeg", "png"])
    if new_image:
        image_to_process = Image.open(new_image)

if image_to_process:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image_to_process, caption="Original Image", use_container_width=True)
    
    with col2:
        st.subheader("Adjust Image Properties")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
        sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0)
        saturation = st.slider("Saturation", 0.5, 2.0, 1.0)
        hue = st.slider("Hue Adjustment", -0.5, 0.5, 0.0)
        
        enhancer = ImageEnhance.Brightness(image_to_process)
        processed_img = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(processed_img)
        processed_img = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Sharpness(processed_img)
        processed_img = enhancer.enhance(sharpness)
        enhancer = ImageEnhance.Color(processed_img)
        processed_img = enhancer.enhance(saturation)
        
        st.image(processed_img, caption="Processed Image", use_container_width=True)
