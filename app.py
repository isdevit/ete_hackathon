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

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Set page configuration
st.set_page_config(page_title="APP HACKATHON 2025", layout="wide")

# Custom Background Video Styling
video_path = "background.mp4"  # Ensure this file is in the working directory
if os.path.exists(video_path):
    st.video(video_path)

# Sidebar
st.sidebar.title("APP HACKATHON 2025 Dashboard")

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
        fig, ax = plt.subplots(figsize=(4,2))  # Reduced size

        if option == "Domain Distribution":
            sns.countplot(data=df, x="Domain", palette="viridis", ax=ax)
            plt.xticks(rotation=45)
            ax.set_title("Domain Distribution")  # Added title

        elif option == "Age Distribution":
            sns.histplot(df["Age"], bins=10, kde=True, color="blue", ax=ax)
            ax.set_title("Age Distribution")  # Added title

        elif option == "Experience vs Rating":
            sns.scatterplot(data=df, x="Experience (Years)", y="Rating (Out of 5)", hue="Domain", palette="deep", ax=ax)
            ax.set_title("Experience vs Rating")  # Added title

        elif option == "Word Cloud (Feedback)":
            text = " ".join(df["Feedback"].astype(str))
            wordcloud = WordCloud(width=500, height=300, background_color='white').generate(text)
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            ax.set_title("Feedback Word Cloud")  # Added title

        st.pyplot(fig)


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
            st.image(uploaded_image, caption=f"Uploaded: {uploaded_image.name}", use_column_width=True)
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
        st.image(image_to_process, caption="Original Image", use_column_width=True)
    
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
        
        st.image(processed_img, caption="Processed Image", use_column_width=True)
