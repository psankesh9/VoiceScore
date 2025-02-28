import os
import re
import requests
import time
import logging
import nltk
from tqdm import tqdm
from bs4 import BeautifulSoup
from transformers import pipeline
import pyttsx3

# Ensure NLTK tokenizer is downloaded
nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to scrape a web novel page
def scrape_page(url):
    start_time = time.time()
    logging.info(f"Starting web scraping for: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP issues
    except requests.RequestException as e:
        logging.error(f"Failed to fetch webpage: {e}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text content from the page
    paragraphs = soup.find_all('p')
    if not paragraphs:
        logging.warning("No paragraph elements found. The website structure might have changed.")
        return None

    text = "\n".join([p.get_text() for p in tqdm(paragraphs, desc="Extracting text", unit="p")])

    # Calculate execution time
    elapsed_time = time.time() - start_time
    logging.info(f"Scraping completed in {elapsed_time:.2f} seconds.")

    return text

# Save extracted text to a file
def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    logging.info(f"Saved scraped content to {filename}")

# Named Entity Recognition (NER) function
def extract_names(text):
    logging.info("Starting Named Entity Recognition (NER)...")
    
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    entities = ner_pipeline(text)

    recognized_names = {}
    for entity in entities:
        if entity.get("entity_group") == "PER":
            name = entity["word"]
            if name not in recognized_names:
                recognized_names[name] = len(recognized_names) + 2  # Voice ID starts from 2

    logging.info(f"NER completed. Found {len(recognized_names)} unique named entities.")
    return recognized_names

# Assign voices to recognized names
def assign_voice_to_sentence(sentence, names_voice_map):
    for name, voice_id in names_voice_map.items():
        if re.search(r'\b' + re.escape(name) + r'\b', sentence, re.IGNORECASE):
            return voice_id
    return 1  # Narrator voice

# Text-to-Speech function with progress tracking
def speak_sentence(engine, sentence, voice_id, voices):
    voice_index = (voice_id - 1) % len(voices)
    engine.setProperty('voice', voices[voice_index].id)
    
    logging.info(f"Using Voice {voice_id} to say: {sentence[:50]}...")  # Shortened sentence preview

    try:
        engine.say(sentence)
        engine.runAndWait()
    except Exception as e:
        logging.error(f"Text-to-Speech error: {e}")

# Main pipeline execution
if __name__ == '__main__':
    url = "https://readnovelfull.com/absolute-resonance/chapter-1481-four-sword-realm-goddess-divineflame-mirror.html"

    total_start_time = time.time()
    
    try:
        # Scrape and save page content
        scraped_text = scrape_page(url)
        if scraped_text:
            save_text_to_file(scraped_text, "scraped_page.txt")

            # Segment text into sentences
            sentences = nltk.sent_tokenize(scraped_text)
            logging.info(f"Segmented text into {len(sentences)} sentences.")

            # Extract Named Entities
            names_voice_map = extract_names(scraped_text)

            if names_voice_map:
                logging.info("Recognized Names and their Voice IDs:")
                for name, voice_id in names_voice_map.items():
                    logging.info(f"  {name}: Voice {voice_id}")

            # Initialize TTS engine
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if not voices:
                logging.error("No TTS voices available. Check pyttsx3 installation.")
                exit()

            # Speak each sentence with the appropriate voice
            for sentence in tqdm(sentences, desc="Generating Speech", unit="sentence"):
                assigned_voice = assign_voice_to_sentence(sentence, names_voice_map)
                speak_sentence(engine, sentence, assigned_voice, voices)

        else:
            logging.error("Scraped text is empty. Check the website structure.")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")

    total_elapsed_time = time.time() - total_start_time
    logging.info(f"Total execution time: {total_elapsed_time:.2f} seconds.")
