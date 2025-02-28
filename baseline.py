import os
import re
import requests
from bs4 import BeautifulSoup
import nltk
from transformers import pipeline
import pyttsx3

# Ensure that NLTK's sentence tokenizer data is downloaded
nltk.download('punkt')

# ---------- Step 1: Scraping the webpage ---------- #
def scrape_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching page: Status code {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract text from the whole page. You may need to adjust this selector.
    text = soup.get_text(separator='\n')
    return text

def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved scraped content to {filename}")

# ---------- Step 2: Segmenting text into sentences ---------- #
def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# ---------- Step 3: Named Entity Recognition (NER) ---------- #
def extract_names(text):
    # Load the NER pipeline with aggregation to combine tokens of full names
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    entities = ner_pipeline(text)
    # Build a mapping of unique person names to a unique voice ID
    # (Here, we reserve Voice 1 for the narrator; person names start at Voice 2)
    recognized_names = {}
    for entity in entities:
        if entity.get("entity_group") == "PER":
            name = entity["word"]
            if name not in recognized_names:
                # Assign voice id starting at 2 (narrator is 1)
                recognized_names[name] = len(recognized_names) + 2
    return recognized_names

# ---------- Step 4: TTS with voice assignment ---------- #
def speak_sentence(engine, sentence, voice_id, voices):
    # Select a voice based on voice_id (cycle if not enough voices)
    voice_index = (voice_id - 1) % len(voices)
    engine.setProperty('voice', voices[voice_index].id)
    print(f"Using Voice {voice_id} to say: {sentence}")
    engine.say(sentence)
    engine.runAndWait()

def assign_voice_to_sentence(sentence, names_voice_map):
    """
    Check if any recognized name from the map appears in the sentence.
    Returns the corresponding voice id if a match is found; otherwise, returns 1 (narrator).
    """
    # For robustness, do a case-insensitive search.
    for name, voice_id in names_voice_map.items():
        # Use regex word boundaries to avoid partial matches.
        if re.search(r'\b' + re.escape(name) + r'\b', sentence, re.IGNORECASE):
            return voice_id
    return 1  # Narrator voice

# ---------- Main Process ---------- #
if __name__ == '__main__':
    # URL to scrape (provided by the user)
    url = "https://readnovelfull.com/absolute-resonance/chapter-1481-four-sword-realm-goddess-divineflame-mirror.html"
    
    try:
        # Scrape and save page content
        full_text = scrape_page(url)
        save_text_to_file(full_text, "scraped_page.txt")
        
        # Segment the full text into sentences
        sentences = get_sentences(full_text)
        print(f"Total sentences extracted: {len(sentences)}")
        
        # Extract recognized person names from the entire text
        names_voice_map = extract_names(full_text)
        print("Recognized Names and their Assigned Voice IDs:")
        for name, vid in names_voice_map.items():
            print(f"  {name}: Voice {vid}")
        
        # Initialize TTS engine (pyttsx3)
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if not voices:
            raise Exception("No voices found in TTS engine.")
        
        # ---------- Step 5: Process each sentence ---------- #
        for sentence in sentences:
            # Determine which voice to use for this sentence
            assigned_voice = assign_voice_to_sentence(sentence, names_voice_map)
            speak_sentence(engine, sentence, assigned_voice, voices)
    
    except Exception as e:
        print(f"An error occurred: {e}")

# ---------- Flowchart of the Logic ----------
"""
[Start]
   │
   ▼
[Scrape webpage using the provided URL]
   │
   ▼
[Extract page content with BeautifulSoup]
   │
   ▼
[Save the scraped text to a .txt file]
   │
   ▼
[Segment the full text into sentences (using NLTK)]
   │
   ▼
[Run NER (Hugging Face transformer) on the full text]
   │
   ▼
[Build a mapping of recognized person names to unique voice IDs]
   │
   ▼
[For each sentence in the text]
   │
   ├──► [Check if the sentence contains any recognized name]
   │         │
   │         ├─ Yes: Assign the corresponding voice (e.g., Voice 3 for "Li Luo")
   │         │
   │         └─ No: Assign narrator voice (Voice 1)
   │
   ▼
[Use the TTS engine (pyttsx3) to speak the sentence using the assigned voice]
   │
   ▼
[Repeat for all sentences]
   │
   ▼
[End]
"""
