import joblib
from sklearn.metrics import classification_report, accuracy_score
import spacy
import os

# Load your trained model
model = joblib.load("models/figurative_speech_model.pkl")

# Load the SpaCy model for text preprocessing (use the same model you used for training)
nlp = spacy.load("en_core_web_sm")  # Use the same SpaCy model

def preprocess(text):
    """
    Preprocess the text by tokenizing, removing stopwords, and lemmatizing.
    This should match the preprocessing used during training.
    """
    doc = nlp(text)
    # Tokenize, remove stopwords, and lemmatize the text
    processed_text = ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return processed_text

# Define your test data with the new text
test_data = [
    "The city at dusk was a canvas, painted with shades of orange and pink as the sun dipped below the horizon.",
    "The streets, like rivers of light, flowed with the cars, their headlights shining like stars in the early evening sky.",
    "People moved through the city as if they were actors on a stage, each with their own story to tell, yet all part of a greater performance.",
    "The buildings stood tall and proud, like sentinels guarding the heart of the urban jungle, casting long shadows over the bustling streets below.",
    "The wind whispered through the trees in the park, a soft, comforting sound that seemed to tell secrets only the night could understand.",
    "As I walked, the pavement beneath my feet felt alive, pulsing with the rhythm of the city, as if it were breathing along with me.",
    "The streetlights flickered on one by one, their glow filling the air like the first notes of a symphony.",
    "It was as though the city itself had come to life, awake and humming with energy.",
    "The moon appeared, shy and silver, a lone traveler in the vast expanse.",
    "The stars twinkled like distant memories, reminding me of the vastness of the universe and my small place in it.",
    "The night embraced the city with a cool, gentle hand, as if to soothe the frenetic pace of the day.",
    "The world around me seemed to pause, breathing in the quiet, before it all started again."
]


# True labels for the test data (these are for evaluation purposes)
true_labels = [
    "Personification",   # The city at dusk was a canvas, painted with shades of orange and pink (Metaphor)
    "Simile",            # The streets, like rivers of light, flowed with the cars (Simile)
    "Metaphor",          # People moved through the city as if they were actors on a stage (Metaphor)
    "Simile",            # The buildings stood tall and proud, like sentinels guarding (Simile)
    "Personification",   # The wind whispered through the trees (Personification)
    "Metaphor",          # As I walked, the pavement beneath my feet felt alive (Metaphor)
    "Simile",            # The streetlights flickered on one by one, their glow filling the air like the first notes of a symphony (Simile)
    "Personification",   # It was as though the city itself had come to life (Personification)
    "Metaphor",          # The moon appeared, shy and silver (Metaphor)
    "Simile",            # The stars twinkled like distant memories (Simile)
    "Personification",   # The night embraced the city with a cool, gentle hand (Personification)
    "Personification"    # The world around me seemed to pause, breathing in the quiet (Personification)
]


# Preprocess the test data
processed_test_data = [preprocess(text) for text in test_data]

# Make predictions using the trained model
predictions = model.predict(processed_test_data)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(true_labels, predictions))

print("Accuracy:", accuracy_score(true_labels, predictions))

# Example: Test with a new individual sentence
new_sentence = "The river flows like a ribbon across the land"
processed_sentence = preprocess(new_sentence)  # Preprocess the new sentence
predicted_label = model.predict([processed_sentence])
print(f"Predicted label for the new sentence: {predicted_label[0]}")
