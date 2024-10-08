import os
import re
import logging
from flask import Flask, render_template, request, jsonify, session
import json
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import nltk

# Ensure you have the NLTK stopwords downloaded
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Add this line for session management

# Folder setup
UPLOAD_FOLDER = 'uploads/'
WORDCLOUD_FOLDER = 'static/wordclouds/'  # Store word clouds in static folder for serving
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WORDCLOUD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['WORDCLOUD_FOLDER'] = WORDCLOUD_FOLDER

# Allowed file types
ALLOWED_EXTENSIONS = {'txt'}

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to clean the text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation and non-alphanumeric characters
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stop words
    return text

# Function to generate word cloud and frequency data
def generate_word_cloud(text, filename):
    try:
        cleaned_text = clean_text(text)
        words = cleaned_text.split()
        word_frequencies = dict(Counter(words).most_common(50))

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
        image_path = os.path.join(app.config['WORDCLOUD_FOLDER'], f"{filename}.png")
        wordcloud.to_file(image_path)

        logger.debug(f"Word cloud saved to {image_path}")
        return image_path, word_frequencies
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, None

# Function to generate word frequencies
def generate_word_frequencies(text):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    word_frequencies = dict(Counter(words).most_common(50))
    return word_frequencies

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
        session['uploaded_file'] = filename  # Store filename in session
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
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
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
        logger.error('No file uploaded')
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        logger.debug(f"Looking for file at: {file_path}")
        if not os.path.exists(file_path):
            logger.error('File not found')
            return jsonify({'error': 'File not found'}), 404
        
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        logger.debug('File read successfully, generating word frequencies')
        word_frequencies = generate_word_frequencies(text)

        logger.debug(f"Generated word frequencies: {word_frequencies}")
        return jsonify(word_frequencies), 200
    except Exception as e:
        logger.error(f"Error generating word frequencies: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/result/word_frequencies')
def word_frequencies_result():
    word_frequencies_str = request.args.get('word_frequencies')

    if not word_frequencies_str:
        return "Missing word frequencies", 400

    try:
        word_frequencies = json.loads(word_frequencies_str)
    except json.JSONDecodeError:
        return "Invalid word frequencies format", 400

    # Ensure word_frequencies is a valid dictionary
    if not isinstance(word_frequencies, dict):
        return "Invalid word frequencies format: Expected a dictionary", 400

    return render_template('result_word_frequencies.html', word_frequencies=word_frequencies)

@app.route('/result/word_cloud')
def word_cloud_result():
    image_filename = request.args.get('image_filename')
    if not image_filename:
        return "Missing image filename", 400

    return render_template('result_word_cloud.html', image_filename=image_filename)

if __name__ == "__main__":
    app.run(debug=True)