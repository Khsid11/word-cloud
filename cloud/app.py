from flask import Flask, render_template, request
from collections import Counter
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import base64
from io import BytesIO
import spacy

app = Flask(__name__)

# Load the spaCy English model for Named Entity Recognition and word vectors
nlp = spacy.load("en_core_web_sm")

# Define the uploads folder
UPLOAD_FOLDER = 'uploads'

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Read text from the uploaded file
        text = read_text_file(file_path)

        # Extract named entities from the text
        named_entities = extract_named_entities(text)

        # Tokenize the text into words and named entities
        words = tokenize_text(text)

        # Remove stopwords and lemmatize words
        words = preprocess_text(words)

        # Count co-occurrences
        co_occurrences = count_co_occurrences(words)

        # Generate word cloud
        word_cloud_image = generate_word_cloud(words)

        # Prepare data for rendering
        co_occurrence_table = pd.DataFrame(co_occurrences.items(), columns=['Word Pair', 'Frequency'])
        co_occurrence_table = co_occurrence_table.sort_values(by='Frequency', ascending=False)

        # Render the result template with the data
        return render_template('result.html', co_occurrence_table=co_occurrence_table.to_html(index=False), word_cloud_image=word_cloud_image, named_entities=', '.join(named_entities))
    
    # Render the index template for GET requests
    return render_template('index.html')

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to tokenize text
def tokenize_text(text):
    words = word_tokenize(text)
    return words

# Function to remove stopwords and lemmatize words
def preprocess_text(words):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
    return processed_words

# Function to count co-occurrences of words with adjustable window size
def count_co_occurrences(words, window_size=6):
    co_occurrences = Counter()
    for i, word1 in enumerate(words):
        for j in range(i + 1, min(i + window_size, len(words))):
            word2 = words[j]
            if word1 != word2:
                co_occurrences[(word1, word2)] += 1
    return co_occurrences

# Function to generate word cloud
def generate_word_cloud(words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    img = wordcloud.to_image()
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Function to extract named entities using spaCy
def extract_named_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
    return entities

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
