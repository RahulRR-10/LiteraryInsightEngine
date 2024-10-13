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
from flask import Flask, request, jsonify, session, send_file, send_from_directory
from transformers import pipeline
from transformers import AutoTokenizer
import random
import openai
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import json
import networkx as nx
import plotly.graph_objects as go
from dotenv import load_dotenv
import traceback 
from googletrans import Translator
from flask import jsonify, session
import os
import networkx as nx
import plotly.graph_objs as go
import spacy
from logging import getLogger

app = Flask(__name__)

app.config['SESSION_COOKIE_SAMESITE'] = 'None'  # Allow cookies to be sent in third-party contexts
app.config['SESSION_COOKIE_SECURE'] = True  # Use secure cookies if your app is served over HTTPS
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-35-turbo")


openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-05-01-preview"  # Update this to the latest API version
openai.api_key = AZURE_OPENAI_KEY


# Initialize SpaCy with only the necessary components
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler", "ner"])
nlp.add_pipe("entity_ruler").from_disk("entity_patterns.jsonl")
nlp.add_pipe('sentencizer', first=True) 
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
app.secret_key = '123'  # Replace with a secure key

app = Flask(__name__)
app.secret_key = os.environ.get('123', 'fallback_secret_key')

# Folder setup
IMAGE_FOLDER = 'static/images/'
os.makedirs(IMAGE_FOLDER, exist_ok=True)
UPLOAD_FOLDER = 'uploads/'
TRANSLATED_FOLDER = 'translated/'
WORDCLOUD_FOLDER = 'static/wordclouds/'
MAPS_FOLDER = 'static/maps/'
for folder in [UPLOAD_FOLDER, WORDCLOUD_FOLDER, MAPS_FOLDER, TRANSLATED_FOLDER]:
    os.makedirs(folder, exist_ok=True)
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    WORDCLOUD_FOLDER=WORDCLOUD_FOLDER,
    MAPS_FOLDER=MAPS_FOLDER,
    IMAGE_FOLDER=IMAGE_FOLDER,
    TRANSLATED_FOLDER=TRANSLATED_FOLDER
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
    

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Your name is LitBot and you are a helpful literary assistant and will provide the user with help in literature."},
                {"role": "user", "content": user_message}
            ]
        )
        
        bot_response = response.choices[0].message['content']
        return jsonify({"response": bot_response})
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request."}), 500
    
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    target_language = data.get('language')

    filename = session.get('uploaded_file')
    if not filename or not target_language:
        return jsonify({'error': 'Filename and language are required'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        return jsonify({'error': 'Uploaded file not found'}), 404
    except Exception as e:
        app.logger.error(f"Error reading file: {str(e)}")
        return jsonify({'error': 'Error reading uploaded file'}), 500

    # Perform translation
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text

    # Save the translated text to a new file
    translated_filename = f'translated_{filename}'
    translated_filepath = os.path.join(app.config['TRANSLATED_FOLDER'], translated_filename)
    
    with open(translated_filepath, 'w', encoding='utf-8') as file:
        file.write(translated_text)

    return jsonify({
        'translated_filename': translated_filename,
        'translated_text': translated_text
    }), 200


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
    
@app.route('/generate_zipf', methods=['POST'])
def generate_zipf_analysis():
    try:
        filename = session.get('uploaded_file')
        if not filename:
            return jsonify({'error': 'No file uploaded'}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {filename}'}), 404

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        cleaned_text = clean_text(text)
        word_frequencies = Counter(cleaned_text.split())

        ranked_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
        zipf_data = {str(rank + 1): freq for rank, (word, freq) in enumerate(ranked_words)}

        session['zipf_data'] = json.dumps(zipf_data)

        return jsonify({'message': 'Zipf analysis completed'}), 200
    except Exception as e:
        error_message = f"Error generating Zipf's Law analysis: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return jsonify({'error': error_message}), 500
    

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
    



@app.route('/check_uploaded_file', methods=['GET'])
def check_uploaded_file():
    filename = session.get('uploaded_file')
    if filename and os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        return jsonify({'file_uploaded': True})
    else:
        return jsonify({'file_uploaded': False})

@app.route('/generate_sentiment', methods=['POST'])
def generate_sentiment():
    try:
        filename = session.get('uploaded_file')
        if not filename:
            return jsonify({'error': 'No file uploaded'}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        if not text:
            return jsonify({'error': 'File is empty'}), 400

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
    



nlp = spacy.load("en_core_web_sm")
logger = getLogger(__name__)

@app.route('/generate_character_relationships', methods=['POST'])
def generate_character_relationships():
    try:
        filename = session.get('uploaded_file')
        if not filename:
            return jsonify({'error': 'No file uploaded'}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404    
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        doc = nlp(text)
        
        # Extract named entities recognized as persons
        persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        logger.debug(f"Persons found: {persons}")
        
        if not persons:
            return jsonify({"error": "No persons found in the text."}), 400
        
        # Create a simple co-occurrence graph
        G = nx.Graph()
        for i, person1 in enumerate(persons):
            for person2 in persons[i+1:]:
                G.add_edge(person1, person2, weight=G.get_edge_data(person1, person2, {'weight': 0})['weight'] + 1)
        
        # Create a Plotly figure
        pos = nx.spring_layout(G)
        edge_trace, node_trace = create_plotly_traces(G, pos)
        
        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0,l=0,r=0,t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        # Save the figure
        plot_filename = 'character_relationships.html'
        plot_path = os.path.join('static', plot_filename)
        fig.write_html(plot_path)
        
        return jsonify({"success": True, "plot_filename": plot_filename})
    
    except Exception as e:
        logger.error(f"Error generating character relationships: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

def create_plotly_traces(G, pos):
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x, node_y = zip(*pos.values())
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        ),
        text=list(pos.keys()),
        textposition="top center"
    )
    
    node_adjacencies = [len(list(G.neighbors(node))) for node in G.nodes()]
    node_trace.marker.color = node_adjacencies
    node_trace.marker.size = [v * 5 for v in node_adjacencies]
    
    return edge_trace, node_trace




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


@app.route('/result/character_relationships')
def character_relationships_result():
    plot_filename = request.args.get('plot_filename')
    if not plot_filename:
        return "Missing plot filename", 400
    
    # Validate the filename to prevent potential security issues
    if not plot_filename.endswith('.html') or '/' in plot_filename:
        return "Invalid plot filename", 400
    
    # Check if the file exists
    plot_path = os.path.join('static', plot_filename)
    if not os.path.exists(plot_path):
        return "Plot file not found", 404
    
    return render_template('result_character.html', plot_filename=plot_filename)

@app.route('/result/zipf')
def zipf_result():
    zipf_data_json = session.get('zipf_data')
    if not zipf_data_json:
        return "No Zipf's data found in session", 400
    
    try:
        zipf_data = json.loads(zipf_data_json)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding Zipf's data from session: {str(e)}")
        return "Invalid Zipf's data format in session", 400

    return render_template('result_zipf.html', zipf_data=zipf_data)


@app.route('/result/translation', methods=['GET'])
def translation_page():
    # This renders the HTML page where the user selects the translation language
    return render_template('result_translate.html')

# Route to process the translation (POST request)
@app.route('/result/translation', methods=['POST'])
def translation_result():
    try:
        data = request.get_json()
        target_language = data.get('language')

        if not target_language:
            return jsonify({'error': 'Target language not provided'}), 400

        filename = session.get('uploaded_file')
        if not filename:
            return jsonify({'error': 'No file uploaded'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Uploaded file not found'}), 404

        # Reading original text from file
        with open(filepath, 'r', encoding='utf-8') as file:
            original_text = file.read()

        # Perform translation
        translator = Translator()
        translated = translator.translate(original_text, dest=target_language)

        # Save the translated text to a new file
        translated_filename = f'translated_{filename}'
        translated_filepath = os.path.join(TRANSLATED_FOLDER, translated_filename)

        with open(translated_filepath, 'w', encoding='utf-8') as file:
            file.write(translated.text)

        # Send back a success response with file download path
        return jsonify({
            'translated_text': translated.text,
            'translated_filename': translated_filename
        }), 200

    except Exception as e:
        app.logger.error(f"Translation error: {str(e)}")
        return jsonify({'error': 'An error occurred during translation'}), 500

# Route to allow file download
@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_from_directory(TRANSLATED_FOLDER, filename, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error sending file: {str(e)}")
        return jsonify({'error': 'Error downloading the file'}), 500




if __name__ == '__main__':
    app.run(debug=True)