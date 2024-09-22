import os
import re
import logging
from flask import Flask, render_template, request, jsonify, session
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import nltk
import folium
from geopy.geocoders import Nominatim
import spacy
from cachetools import TTLCache
from functools import lru_cache

# Initialize SpaCy
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

# Initialize Nominatim geolocator with caching
geolocator = Nominatim(user_agent="geoapiExercises")
geocode_cache = TTLCache(maxsize=1000, ttl=86400)

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
    return list(set(ent.text for ent in doc.ents if ent.label_ == "GPE"))

def geocode_place(place):
    if place in geocode_cache:
        return geocode_cache[place]
    
    try:
        location = geolocator.geocode(place)
        if location:
            result = (location.latitude, location.longitude)
            geocode_cache[place] = result
            return result
    except Exception as e:
        logger.error(f"Geocoding error for '{place}': {str(e)}")
    return None

def generate_map(locations, filename):
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    for location in locations:
        lat_lon = geocode_place(location)
        if lat_lon:
            folium.Marker(lat_lon, popup=location).add_to(m)
        else:
            logger.warning(f"Location '{location}' could not be geocoded.")

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
        if not locations:
            return jsonify({'message': 'No locations found in the text'}), 200

        map_path = generate_map(locations, filename)
        return jsonify({'map_filename': os.path.basename(map_path)}), 200
    except Exception as e:
        logger.error(f"Error generating geospatial data: {str(e)}")
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    app.run(debug=False)