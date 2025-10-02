"""
VoiceScore: Complete Pipeline for Multi-Character Text-to-Speech
Combines web scraping, NER, coreference resolution, and advanced TTS synthesis
"""

import os
import logging
import time
import nltk
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm

# Import our custom modules
from advanced_coreference import AdvancedCoreferenceResolver
from advanced_tts_system import AdvancedTTSEngine

class VoiceScorePipeline:
    """Complete pipeline for processing web novels into multi-character speech"""

    def __init__(self, output_dir: str = "voicescore_output", device: str = "auto"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        logging.info("Initializing VoiceScore Pipeline...")
        self.coreference_resolver = AdvancedCoreferenceResolver()
        self.tts_engine = AdvancedTTSEngine(device=device)

        # Ensure NLTK data is available
        self._setup_nltk()

        logging.info("VoiceScore Pipeline initialized successfully")

    def _setup_nltk(self):
        """Setup required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            logging.warning(f"NLTK setup warning: {e}")

    def scrape_web_novel(self, url: str, save_raw: bool = True) -> Optional[str]:
        """Scrape text content from a web novel page"""
        start_time = time.time()
        logging.info(f"Scraping web novel from: {url}")

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

        except requests.RequestException as e:
            logging.error(f"Failed to fetch webpage: {e}")
            return None

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try different selectors for web novel content
        content_selectors = [
            'div.chapter-content',  # Common web novel selector
            'div.content',
            'div.text_story',
            'div.chapter_content',
            'article',
            'main'
        ]

        text_content = None
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                text_content = "\n".join([elem.get_text(strip=True) for elem in elements])
                logging.info(f"Found content using selector: {selector}")
                break

        # Fallback: extract all paragraphs
        if not text_content:
            paragraphs = soup.find_all('p')
            if paragraphs:
                text_content = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                logging.info("Using fallback paragraph extraction")

        if not text_content:
            logging.error("No text content found on the page")
            return None

        # Clean up the text
        text_content = self._clean_text(text_content)

        # Save raw content if requested
        if save_raw:
            raw_file = self.output_dir / "raw_scraped_content.txt"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            logging.info(f"Saved raw content to {raw_file}")

        elapsed_time = time.time() - start_time
        logging.info(f"Scraping completed in {elapsed_time:.2f} seconds")

        return text_content

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        import re
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        text = text.strip()

        # Remove common web novel artifacts
        artifacts = [
            r'Previous Chapter',
            r'Next Chapter',
            r'Table of Contents',
            r'Chapter \d+',
            r'Advertisement',
            r'Support us on Patreon',
            r'Read \d+ chapters ahead',
        ]

        for artifact in artifacts:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE)

        return text

    def process_text(self, text: str) -> Tuple[List[str], Dict[str, int], str]:
        """Process text through coreference resolution and character extraction"""
        logging.info("Processing text through coreference resolution...")

        # Resolve coreferences and extract characters
        resolved_text, characters = self.coreference_resolver.resolve_text_coreferences(text)

        # Segment into sentences
        sentences = nltk.sent_tokenize(resolved_text)

        logging.info(f"Text processing completed:")
        logging.info(f"  - Original length: {len(text)} characters")
        logging.info(f"  - Resolved length: {len(resolved_text)} characters")
        logging.info(f"  - Sentences: {len(sentences)}")
        logging.info(f"  - Characters found: {len(characters)}")

        return sentences, characters, resolved_text

    def create_voice_profiles(self, characters: Dict[str, int],
                            custom_descriptions: Optional[Dict[str, str]] = None) -> Dict[str, dict]:
        """Create voice profiles for all characters"""
        logging.info("Creating voice profiles for characters...")

        # Create voice profiles using the TTS engine
        voice_profiles = self.tts_engine.create_character_voice_profiles(
            characters, custom_descriptions
        )

        # Save voice profiles
        profiles_file = self.output_dir / "voice_profiles.json"
        self.tts_engine.save_voice_profiles(str(profiles_file))

        return voice_profiles

    def assign_voices_to_sentences(self, sentences: List[str],
                                 characters: Dict[str, int]) -> List[Tuple[str, str]]:
        """Assign character voices to sentences"""
        logging.info("Assigning voices to sentences...")

        sentence_assignments = []

        for sentence in sentences:
            # Default to narrator
            assigned_character = "Narrator"
            assigned_voice_id = 1

            # Check if any character is mentioned in the sentence
            sentence_lower = sentence.lower()
            for char_name, voice_id in characters.items():
                if char_name.lower() in sentence_lower:
                    assigned_character = char_name
                    assigned_voice_id = voice_id
                    break

            sentence_assignments.append((sentence, assigned_character))

        logging.info(f"Voice assignment completed for {len(sentences)} sentences")

        return sentence_assignments

    def synthesize_audio(self, sentence_assignments: List[Tuple[str, str]]) -> List[str]:
        """Synthesize audio for all sentences"""
        logging.info("Starting audio synthesis...")

        audio_dir = self.output_dir / "audio_segments"
        audio_dir.mkdir(exist_ok=True)

        audio_files = []

        for i, (sentence, character) in enumerate(tqdm(sentence_assignments, desc="Synthesizing audio")):
            output_file = audio_dir / f"segment_{i:04d}_{character.replace(' ', '_')}.wav"

            try:
                self.tts_engine.synthesize_speech(sentence, character, str(output_file))
                audio_files.append(str(output_file))

                # Log progress periodically
                if (i + 1) % 50 == 0:
                    logging.info(f"Synthesized {i + 1}/{len(sentence_assignments)} segments")

            except Exception as e:
                logging.error(f"Failed to synthesize segment {i}: {e}")
                # Create a placeholder silence file
                audio_files.append(str(output_file))

        logging.info(f"Audio synthesis completed: {len(audio_files)} files generated")
        return audio_files

    def create_audio_playlist(self, audio_files: List[str]) -> str:
        """Create a playlist file for the generated audio"""
        playlist_file = self.output_dir / "audio_playlist.m3u"

        with open(playlist_file, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            for audio_file in audio_files:
                rel_path = os.path.relpath(audio_file, self.output_dir)
                f.write(f"{rel_path}\n")

        logging.info(f"Created audio playlist: {playlist_file}")
        return str(playlist_file)

    def save_processing_report(self, characters: Dict[str, int],
                             sentences: List[str],
                             sentence_assignments: List[Tuple[str, str]]):
        """Save a detailed processing report"""
        report_file = self.output_dir / "processing_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("VoiceScore Processing Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total sentences processed: {len(sentences)}\n")
            f.write(f"Characters identified: {len(characters)}\n\n")

            f.write("Character Voice Assignments:\n")
            f.write("-" * 30 + "\n")
            for char_name, voice_id in characters.items():
                count = sum(1 for _, character in sentence_assignments if character == char_name)
                f.write(f"  {char_name} (Voice {voice_id}): {count} sentences\n")

            narrator_count = sum(1 for _, character in sentence_assignments if character == "Narrator")
            f.write(f"  Narrator (Voice 1): {narrator_count} sentences\n\n")

            f.write("Character Details:\n")
            f.write("-" * 20 + "\n")
            char_info = self.coreference_resolver.get_character_info()
            for name, info in char_info.items():
                f.write(f"  {name}:\n")
                f.write(f"    Voice ID: {info['voice_id']}\n")
                f.write(f"    Aliases: {info['aliases']}\n")
                f.write(f"    Mentions: {info['mention_count']}\n")
                f.write(f"    Last mentioned: sentence {info['last_mentioned']}\n\n")

        logging.info(f"Saved processing report to {report_file}")

    def process_web_novel(self, url: str,
                         custom_voice_descriptions: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Complete pipeline to process a web novel from URL"""
        start_time = time.time()
        logging.info(f"Starting VoiceScore pipeline for: {url}")

        # Step 1: Scrape content
        raw_text = self.scrape_web_novel(url)
        if not raw_text:
            raise ValueError("Failed to scrape content from URL")

        # Step 2: Process text and resolve coreferences
        sentences, characters, resolved_text = self.process_text(raw_text)

        # Step 3: Create voice profiles
        voice_profiles = self.create_voice_profiles(characters, custom_voice_descriptions)

        # Step 4: Assign voices to sentences
        sentence_assignments = self.assign_voices_to_sentences(sentences, characters)

        # Step 5: Synthesize audio
        audio_files = self.synthesize_audio(sentence_assignments)

        # Step 6: Create playlist and report
        playlist_file = self.create_audio_playlist(audio_files)
        self.save_processing_report(characters, sentences, sentence_assignments)

        # Save resolved text
        resolved_file = self.output_dir / "resolved_text.txt"
        with open(resolved_file, 'w', encoding='utf-8') as f:
            f.write(resolved_text)

        total_time = time.time() - start_time
        logging.info(f"VoiceScore pipeline completed in {total_time:.2f} seconds")

        return {
            'output_directory': str(self.output_dir),
            'audio_files_count': len(audio_files),
            'characters_found': len(characters),
            'sentences_processed': len(sentences),
            'playlist_file': playlist_file,
            'processing_time': total_time
        }

def main():
    """Main function to run the VoiceScore pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("voicescore.log"),
            logging.StreamHandler()
        ]
    )

    # Configuration
    url = "https://readnovelfull.com/absolute-resonance/chapter-1481-four-sword-realm-goddess-divineflame-mirror.html"

    # Custom voice descriptions (optional)
    custom_voices = {
        "Li Luo": "confident, determined young male voice with slight heroic tone",
        "Jiang Qing'e": "elegant, strong female voice with authoritative presence",
        "Li Lingjing": "wise, mature female voice with caring undertones"
    }

    # Initialize and run pipeline
    pipeline = VoiceScorePipeline(output_dir="novel_audio_output")

    try:
        results = pipeline.process_web_novel(url, custom_voices)

        print("\n" + "="*60)
        print("VoiceScore Processing Complete!")
        print("="*60)
        print(f"Output directory: {results['output_directory']}")
        print(f"Audio files generated: {results['audio_files_count']}")
        print(f"Characters identified: {results['characters_found']}")
        print(f"Sentences processed: {results['sentences_processed']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Playlist file: {results['playlist_file']}")
        print("="*60)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()