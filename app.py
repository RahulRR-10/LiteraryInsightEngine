import os
import re
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import nltk
import folium
from folium.plugins import MarkerCluster, FastMarkerCluster
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import spacy
from cachetools import TTLCache, LRUCache
from functools import lru_cache
import ujson
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint
import time
from textblob import TextBlob 
import matplotlib.pyplot as plt
import io
import base64
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

import random


app = Flask(__name__)


# Initialize SpaCy with only the necessary components
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler", "ner"])
nlp.add_pipe("entity_ruler").from_disk("entity_patterns.jsonl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Initialize the geocode cache
geocode_cache = TTLCache(maxsize=1000, ttl=86400)

# Initialize Nominatim geolocator with custom adapter and increased timeout
geolocator = Nominatim(
    user_agent="geoapiExercises",
    timeout=5,
    domain='nominatim.openstreetmap.org'
)

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_secret_key')

# Folder setup
UPLOAD_FOLDER = 'uploads/'
WORDCLOUD_FOLDER = 'static/wordclouds/'
MAPS_FOLDER = 'static/maps/'
for folder in [UPLOAD_FOLDER, WORDCLOUD_FOLDER, MAPS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    WORDCLOUD_FOLDER=WORDCLOUD_FOLDER,
    MAPS_FOLDER=MAPS_FOLDER
)

ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@lru_cache(maxsize=100)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)

def generate_word_cloud(text, filename):
    try:
        cleaned_text = clean_text(text)
        word_frequencies = dict(Counter(cleaned_text.split()).most_common(50))

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
        image_path = os.path.join(app.config['WORDCLOUD_FOLDER'], f"{filename}.png")
        wordcloud.to_file(image_path)

        logger.info(f"Word cloud saved to {image_path}")
        return image_path, word_frequencies
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, None

def extract_locations(text):
    doc = nlp(text)
    locations = {ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}
    
    # Additional step to catch country names that might be missed
    potential_countries = set(text.split())  # Split the text into words
    countries = set()
    for loc in potential_countries:
        if len(loc) > 1 and loc.isalpha():  # Basic filtering
            if geocode_place(loc):
                countries.add(loc)
    
    locations.update(countries)
    
    logger.info(f"Extracted locations: {locations}")
    return locations


def geocode_place(place, max_retries=3):
    if place in geocode_cache:
        return geocode_cache[place]
    
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(place, exactly_one=True)
            if location:
                result = (location.latitude, location.longitude)
                geocode_cache[place] = result
                logger.info(f"Geocoded '{place}' to {result}")
                return result
            else:
                logger.warning(f"Could not geocode '{place}'")
                return None
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            logger.warning(f"Geocoding attempt {attempt + 1} failed for '{place}': {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1 + random.random())  # Wait for 1-2 seconds before retrying
    
    logger.error(f"Failed to geocode '{place}' after {max_retries} attempts")
    return None

def generate_map(locations, filename):
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    for location in locations:
        lat_lon = geocode_place(location)
        if lat_lon:
            folium.Marker(lat_lon, popup=location).add_to(m)
        else:
            logger.warning(f"Location '{location}' could not be geocoded.")
        time.sleep(0.1)  # Add a small delay between geocoding requests

    map_path = os.path.join(app.config['MAPS_FOLDER'], f"{filename}.html")
    m.save(map_path)
    logger.info(f"Map saved to {map_path}")
    return map_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['uploaded_file'] = filename
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    filename = session.get('uploaded_file')
    if not filename:
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        image_path, word_frequencies = generate_word_cloud(text, filename)

        if image_path is None:
            return jsonify({'error': 'Word cloud generation failed'}), 500

        return jsonify({
            'image_filename': os.path.basename(image_path),
            'word_frequencies': word_frequencies
        }), 200
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_word_frequencies', methods=['POST'])
def generate_word_frequencies_endpoint():
    filename = session.get('uploaded_file')
    if not filename:
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        cleaned_text = clean_text(text)
        word_frequencies = dict(Counter(cleaned_text.split()).most_common(50))

        return jsonify(word_frequencies), 200
    except Exception as e:
        logger.error(f"Error generating word frequencies: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_geospatial', methods=['POST'])
def generate_geospatial():
    filename = session.get('uploaded_file')
    if not filename:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        locations = extract_locations(text)
        logger.info(f"Extracted locations from {filename}: {locations}")
        
        if not locations:
            return jsonify({'message': 'No locations found in the text'}), 200
        
        map_path = generate_map(locations, filename)
        if map_path:
            return jsonify({'map_filename': os.path.basename(map_path)}), 200
        else:
            return jsonify({'error': 'Failed to generate map due to geocoding errors'}), 500
    except Exception as e:
        logger.error(f"Error generating geospatial data: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

# def generate_sentiment_graph(image_path, sentiment_score):
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(8, 5))
    
#     # Data for the sentiment visualization
#     categories = ['Positive', 'Neutral', 'Negative']
#     values = [0, 0, 0]

#     if sentiment_score > 0:
#         values[0] = sentiment_score  # Positive
#     elif sentiment_score < 0:
#         values[2] = -sentiment_score  # Negative
#     else:
#         values[1] = 1  # Neutral

#     plt.bar(categories, values, color=['rgba(0, 123, 255, 0.6)', 'rgba(255, 193, 7, 0.6)', 'rgba(220, 53, 69, 0.6)'])
#     plt.ylabel('Frequency')
#     plt.title('Sentiment Analysis')
#     plt.savefig(image_path)
#     plt.close()


@app.route('/generate_sentiment', methods=['POST'])
def generate_sentiment():
    try:
        # Check if the request contains JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        text = data.get('text')

        # Check if text is provided and not empty
        if not text or not isinstance(text, str):
            return jsonify({'error': 'Text must be provided and must be a non-empty string'}), 400

        # Generate sentiment score and description
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity  # Score between -1 (negative) and 1 (positive)
        
        # Determine sentiment description
        if sentiment_score > 0:
            sentiment_description = "Positive"
        elif sentiment_score < 0:
            sentiment_description = "Negative"
        else:
            sentiment_description = "Neutral"

        return jsonify({
            'sentiment_score': sentiment_score,
            'sentiment_description': sentiment_description
        })
    except Exception as e:
        logger.error(f"Error in generate_sentiment: {str(e)}")
        return jsonify({'error': str(e)}), 500  # Return a server error response




@app.route('/result/word_frequencies')
def word_frequencies_result():
    word_frequencies = request.args.get('word_frequencies')
    if not word_frequencies:
        return "Missing word frequencies", 400

    try:
        word_frequencies = eval(word_frequencies)
    except:
        return "Invalid word frequencies format", 400

    return render_template('result_word_frequencies.html', word_frequencies=word_frequencies)

@app.route('/result/word_cloud')
def word_cloud_result():
    image_filename = request.args.get('image_filename')
    if not image_filename:
        return "Missing image filename", 400

    return render_template('result_word_cloud.html', image_filename=image_filename)

@app.route('/result/geospatial')
def geospatial_result():
    map_filename = request.args.get('map_filename')
    if not map_filename:
        return "Missing map filename", 400

    map_path = os.path.join(app.config['MAPS_FOLDER'], map_filename)
    if not os.path.exists(map_path):
        return "Map not found", 404

    return render_template('result_geospatial.html', map_filename=map_filename)


@app.route('/result/sentiment')
def sentiment_result():
    sentiment_score = request.args.get('sentiment_score')
    sentiment_description = request.args.get('sentiment_description')

    if sentiment_score is None or sentiment_description is None:
        return "Missing sentiment data", 400

    return render_template('result_sentiment.html', 
                           sentiment_score=sentiment_score, 
                           sentiment_description=sentiment_description)

if __name__ == '__main__':
    app.run(debug=True)