"""
Enhanced VoiceScore: Complete Pipeline with Neural Embeddings and Advanced TTS
Combines enhanced NER, neural coreference resolution, ECAPA-TDNN voice embeddings,
and CLAP text-to-audio matching for superior character voice synthesis
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
import json
import numpy as np

# Import enhanced modules
from enhanced_coreference import EnhancedCoreferenceResolver
from enhanced_tts_system import EnhancedTTSEngine

class EnhancedVoiceScorePipeline:
    """Enhanced pipeline with neural embeddings and advanced voice matching"""

    def __init__(self, output_dir: str = "enhanced_voicescore_output",
                 device: str = "auto",
                 voice_sample_dir: str = "voice_samples",
                 use_neural_coref: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize enhanced components
        logging.info("Initializing Enhanced VoiceScore Pipeline...")

        self.coreference_resolver = EnhancedCoreferenceResolver(use_neural_coref=use_neural_coref)
        self.tts_engine = EnhancedTTSEngine(device=device, voice_sample_dir=voice_sample_dir)

        # Setup NLTK
        self._setup_nltk()

        # Pipeline statistics
        self.stats = {
            'processing_start_time': None,
            'total_processing_time': 0,
            'characters_found': 0,
            'sentences_processed': 0,
            'audio_files_generated': 0,
            'coreference_resolutions': 0,
            'embedding_matches': 0
        }

        logging.info("Enhanced VoiceScore Pipeline initialized successfully")

    def _setup_nltk(self):
        """Setup required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logging.warning(f"NLTK setup warning: {e}")

    def scrape_web_novel_enhanced(self, url: str, save_raw: bool = True) -> Optional[str]:
        """Enhanced web scraping with better content extraction"""
        start_time = time.time()
        logging.info(f"Enhanced scraping from: {url}")

        try:
            # Enhanced headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }

            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()

        except requests.RequestException as e:
            logging.error(f"Enhanced scraping failed: {e}")
            return None

        # Enhanced content extraction
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Try multiple content selectors (ordered by priority)
        content_selectors = [
            'div.chapter-content',
            'div.content',
            'div.text_story',
            'div.chapter_content',
            'div.reading-content',
            'article.chapter',
            'main.content',
            'div.entry-content',
            'div.post-content',
            'article',
            'main'
        ]

        text_content = None
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                text_parts = []
                for elem in elements:
                    # Extract text while preserving paragraph structure
                    paragraphs = elem.find_all('p')
                    if paragraphs:
                        for p in paragraphs:
                            text = p.get_text(strip=True)
                            if text and len(text) > 10:  # Filter out short noise
                                text_parts.append(text)
                    else:
                        # Fallback to direct text extraction
                        text = elem.get_text(strip=True)
                        if text and len(text) > 50:
                            text_parts.append(text)

                if text_parts:
                    text_content = "\n\n".join(text_parts)
                    logging.info(f"✓ Content extracted using selector: {selector}")
                    break

        # Enhanced fallback: extract all meaningful paragraphs
        if not text_content:
            paragraphs = soup.find_all('p')
            if paragraphs:
                text_parts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 20 and not self._is_noise_text(text):
                        text_parts.append(text)

                if text_parts:
                    text_content = "\n\n".join(text_parts)
                    logging.info("✓ Content extracted using enhanced paragraph fallback")

        if not text_content:
            logging.error("❌ No meaningful content found on the page")
            return None

        # Enhanced text cleaning
        text_content = self._clean_text_enhanced(text_content)

        # Save raw content with metadata
        if save_raw:
            raw_file = self.output_dir / "enhanced_raw_content.txt"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"URL: {url}\n")
                f.write(f"Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Content length: {len(text_content)} characters\n")
                f.write("=" * 80 + "\n\n")
                f.write(text_content)
            logging.info(f"✓ Enhanced raw content saved to {raw_file}")

        elapsed_time = time.time() - start_time
        logging.info(f"Enhanced scraping completed in {elapsed_time:.2f} seconds")

        return text_content

    def _is_noise_text(self, text: str) -> bool:
        """Identify noise text that should be filtered out"""
        noise_patterns = [
            r'^Advertisement$',
            r'^Next Chapter$',
            r'^Previous Chapter$',
            r'^Table of Contents$',
            r'^Chapter \d+$',
            r'^Read \d+ chapters? ahead',
            r'^Support.*Patreon',
            r'^Join.*Discord',
            r'^\d+$',  # Just numbers
            r'^[<>]+$',  # Just brackets
            r'^\s*$'  # Just whitespace
        ]

        import re
        for pattern in noise_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True

        # Filter very short text
        if len(text.strip()) < 10:
            return True

        # Filter text that's mostly special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.5:
            return True

        return False

    def _clean_text_enhanced(self, text: str) -> str:
        """Enhanced text cleaning and normalization"""
        import re

        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines → double line break
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs → single space
        text = re.sub(r'\n ', '\n', text)  # Remove spaces after line breaks

        # Remove common web artifacts
        artifacts = [
            r'Previous Chapter\s*(?:\||\n)',
            r'Next Chapter\s*(?:\||\n)',
            r'Table of Contents\s*(?:\||\n)',
            r'Chapter \d+:?\s*\n',
            r'Advertisement\s*\n',
            r'Support us on Patreon.*?\n',
            r'Read \d+ chapters? ahead.*?\n',
            r'Join.*?Discord.*?\n',
            r'\[.*?\]',  # Remove content in square brackets
            r'www\.\S+',  # Remove URLs
            r'https?://\S+',  # Remove full URLs
        ]

        for artifact in artifacts:
            text = re.sub(artifact, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # Clean up quotation marks
        text = re.sub(r'"([^"]*)"', r'"\1"', text)  # Normalize quotes
        text = re.sub(r''([^']*)' ', r'"\1" ', text)  # Convert smart quotes

        # Normalize dialogue formatting
        text = re.sub(r'\n"([^"]*)"', r'\n"\1"', text)

        return text.strip()

    def process_text_enhanced(self, text: str) -> Tuple[List[str], Dict[str, int], str, Dict]:
        """Enhanced text processing with neural coreference resolution"""
        logging.info("Starting enhanced text processing...")
        start_time = time.time()

        # Enhanced coreference resolution
        resolved_text, characters = self.coreference_resolver.resolve_text_coreferences(text)

        # Get enhanced character information
        character_analysis = self.coreference_resolver.get_enhanced_character_info()

        # Segment into sentences
        sentences = nltk.sent_tokenize(resolved_text)

        # Count coreference resolutions
        original_sentences = nltk.sent_tokenize(text)
        coreference_count = sum(1 for orig, resolved in zip(original_sentences, sentences)
                              if orig != resolved)

        processing_time = time.time() - start_time

        # Update statistics
        self.stats['characters_found'] = len(characters)
        self.stats['sentences_processed'] = len(sentences)
        self.stats['coreference_resolutions'] = coreference_count

        logging.info(f"Enhanced text processing completed in {processing_time:.2f}s:")
        logging.info(f"  ✓ Original length: {len(text)} characters")
        logging.info(f"  ✓ Resolved length: {len(resolved_text)} characters")
        logging.info(f"  ✓ Sentences: {len(sentences)}")
        logging.info(f"  ✓ Characters found: {len(characters)}")
        logging.info(f"  ✓ Coreference resolutions: {coreference_count}")

        return sentences, characters, resolved_text, character_analysis

    def create_enhanced_voice_profiles(self, characters: Dict[str, int],
                                     custom_descriptions: Optional[Dict[str, str]] = None,
                                     voice_samples: Optional[Dict[str, str]] = None) -> Dict:
        """Create enhanced voice profiles with neural embeddings"""
        logging.info("Creating enhanced voice profiles...")

        # Create enhanced voice profiles
        voice_profiles = self.tts_engine.create_character_voice_profiles(
            characters, custom_descriptions, voice_samples
        )

        # Save comprehensive voice analysis
        analysis_file = self.output_dir / "enhanced_voice_analysis.json"
        voice_analysis = self.tts_engine.get_voice_analysis_report()

        with open(analysis_file, 'w') as f:
            json.dump(voice_analysis, f, indent=2)

        logging.info(f"✓ Enhanced voice analysis saved to {analysis_file}")

        return voice_profiles

    def assign_voices_enhanced(self, sentences: List[str], characters: Dict[str, int],
                             character_analysis: Dict) -> List[Tuple[str, str, Dict]]:
        """Enhanced voice assignment with context awareness"""
        logging.info("Starting enhanced voice assignment...")

        assignments = []
        speaker_continuity = {}  # Track speaker across sentences
        dialogue_context = []

        for i, sentence in enumerate(sentences):
            # Default assignment
            assigned_character = "Narrator"
            assigned_voice_id = 1
            confidence = 0.5
            assignment_reason = "default"

            # Enhanced character detection in sentence
            sentence_lower = sentence.lower()
            detected_characters = []

            # Check for direct character mentions
            for char_name, voice_id in characters.items():
                if char_name.lower() in sentence_lower:
                    detected_characters.append((char_name, voice_id, "direct_mention"))

            # Check for character aliases
            for char_name, char_info in character_analysis.items():
                for alias in char_info.get('aliases', []):
                    if alias.lower() in sentence_lower and alias.lower() != char_name.lower():
                        detected_characters.append((char_name, characters.get(char_name, 1), "alias_match"))

            # Enhanced dialogue detection
            if '"' in sentence or "'" in sentence:
                # This is likely dialogue
                dialogue_context.append(i)

                # Try to identify speaker from context
                if detected_characters:
                    # Speaker is likely the most important character mentioned
                    char_scores = []
                    for char_name, voice_id, reason in detected_characters:
                        importance = character_analysis.get(char_name, {}).get('importance_score', 0.0)
                        char_scores.append((char_name, voice_id, importance, reason))

                    if char_scores:
                        char_scores.sort(key=lambda x: x[2], reverse=True)  # Sort by importance
                        assigned_character, assigned_voice_id, importance, assignment_reason = char_scores[0]
                        confidence = min(0.9, 0.6 + importance * 0.3)

                # Check speaker continuity in dialogue
                elif len(dialogue_context) > 1 and i - dialogue_context[-2] <= 3:
                    # Recent dialogue, might be same speaker
                    recent_assignment = assignments[dialogue_context[-2]]
                    if recent_assignment[1] != "Narrator":
                        assigned_character = recent_assignment[1]
                        assigned_voice_id = characters.get(assigned_character, 1)
                        assignment_reason = "dialogue_continuity"
                        confidence = 0.7

            else:
                # Narrative text
                if detected_characters:
                    # For narrative, use the most prominent character as context
                    char_scores = [(char_name, voice_id, character_analysis.get(char_name, {}).get('importance_score', 0.0))
                                 for char_name, voice_id, _ in detected_characters]
                    char_scores.sort(key=lambda x: x[2], reverse=True)

                    # Use narrator voice but note the character context
                    assigned_character = "Narrator"
                    assigned_voice_id = 1
                    assignment_reason = f"narrative_about_{char_scores[0][0]}"
                    confidence = 0.6

            # Store assignment with metadata
            assignment_info = {
                'character': assigned_character,
                'voice_id': assigned_voice_id,
                'confidence': confidence,
                'reason': assignment_reason,
                'detected_characters': [char[0] for char in detected_characters],
                'is_dialogue': '"' in sentence or "'" in sentence,
                'sentence_index': i
            }

            assignments.append((sentence, assigned_character, assignment_info))

        logging.info(f"Enhanced voice assignment completed:")
        logging.info(f"  ✓ Total assignments: {len(assignments)}")

        # Count assignment types
        dialogue_count = sum(1 for _, _, info in assignments if info['is_dialogue'])
        narrator_count = sum(1 for _, char, _ in assignments if char == "Narrator")

        logging.info(f"  ✓ Dialogue sentences: {dialogue_count}")
        logging.info(f"  ✓ Narrator sentences: {narrator_count}")

        return assignments

    def synthesize_audio_enhanced(self, assignments: List[Tuple[str, str, Dict]]) -> List[str]:
        """Enhanced audio synthesis with progress tracking"""
        logging.info("Starting enhanced audio synthesis...")
        start_time = time.time()

        audio_dir = self.output_dir / "enhanced_audio_segments"
        audio_dir.mkdir(exist_ok=True)

        audio_files = []
        synthesis_stats = {'success': 0, 'failures': 0, 'silence_generated': 0}

        # Progress tracking
        with tqdm(assignments, desc="Synthesizing enhanced audio", unit="sentence") as pbar:
            for i, (sentence, character, info) in enumerate(pbar):
                # Create descriptive filename
                confidence_str = f"{info['confidence']:.2f}"
                clean_char = character.replace(' ', '_').replace('.', '')
                output_file = audio_dir / f"seg_{i:04d}_{clean_char}_{confidence_str}.wav"

                try:
                    result_file = self.tts_engine.synthesize_speech(
                        sentence, character, str(output_file)
                    )

                    audio_files.append(str(result_file))
                    synthesis_stats['success'] += 1

                    # Update progress description
                    if (i + 1) % 50 == 0:
                        pbar.set_description(f"Synthesizing enhanced audio ({synthesis_stats['success']} success)")

                except Exception as e:
                    logging.error(f"Enhanced synthesis failed for segment {i}: {e}")
                    # Generate silence as fallback
                    self.tts_engine._generate_silence(str(output_file))
                    audio_files.append(str(output_file))
                    synthesis_stats['failures'] += 1

        synthesis_time = time.time() - start_time
        self.stats['audio_files_generated'] = len(audio_files)

        logging.info(f"Enhanced audio synthesis completed in {synthesis_time:.2f}s:")
        logging.info(f"  ✓ Total files: {len(audio_files)}")
        logging.info(f"  ✓ Successful: {synthesis_stats['success']}")
        logging.info(f"  ✓ Failures: {synthesis_stats['failures']}")

        return audio_files

    def create_enhanced_outputs(self, assignments: List[Tuple[str, str, Dict]],
                               audio_files: List[str],
                               character_analysis: Dict) -> Dict[str, str]:
        """Create enhanced output files and reports"""
        outputs = {}

        # Enhanced playlist with metadata
        playlist_file = self.output_dir / "enhanced_audio_playlist.m3u8"
        with open(playlist_file, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            for i, (audio_file, (sentence, character, info)) in enumerate(zip(audio_files, assignments)):
                rel_path = os.path.relpath(audio_file, self.output_dir)
                duration = 3.0  # Estimate, could be calculated from audio
                f.write(f"#EXTINF:{duration},{character} - {info['reason']} (conf:{info['confidence']:.2f})\n")
                f.write(f"{rel_path}\n")
        outputs['playlist'] = str(playlist_file)

        # Comprehensive processing report
        report_file = self.output_dir / "enhanced_processing_report.json"
        report = {
            'pipeline_info': {
                'version': 'Enhanced VoiceScore v2.0',
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_processing_time': self.stats['total_processing_time']
            },
            'statistics': self.stats,
            'characters': {
                name: {
                    **info,
                    'audio_assignments': sum(1 for _, char, _ in assignments if char == name),
                    'dialogue_lines': sum(1 for _, char, meta in assignments
                                        if char == name and meta['is_dialogue']),
                    'narrative_mentions': sum(1 for _, char, meta in assignments
                                            if char == name and not meta['is_dialogue'])
                }
                for name, info in character_analysis.items()
            },
            'assignment_analysis': {
                'total_sentences': len(assignments),
                'dialogue_sentences': sum(1 for _, _, info in assignments if info['is_dialogue']),
                'narrator_sentences': sum(1 for _, char, _ in assignments if char == "Narrator"),
                'average_confidence': np.mean([info['confidence'] for _, _, info in assignments]),
                'assignment_reasons': {
                    reason: sum(1 for _, _, info in assignments if info['reason'] == reason)
                    for reason in set(info['reason'] for _, _, info in assignments)
                }
            },
            'embedding_statistics': self.tts_engine.get_voice_analysis_report()['embedding_stats']
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        outputs['report'] = str(report_file)

        # Detailed assignment log
        assignment_file = self.output_dir / "enhanced_voice_assignments.csv"
        with open(assignment_file, 'w', encoding='utf-8') as f:
            f.write("Sentence_ID,Character,Confidence,Reason,Is_Dialogue,Detected_Characters,Text_Preview\n")
            for i, (sentence, character, info) in enumerate(assignments):
                preview = sentence[:50].replace('"', '""')  # Escape quotes for CSV
                detected = "|".join(info['detected_characters'])
                f.write(f"{i},{character},{info['confidence']:.3f},{info['reason']},{info['is_dialogue']},{detected},\"{preview}...\"\n")
        outputs['assignments'] = str(assignment_file)

        return outputs

    def process_web_novel_enhanced(self, url: str,
                                 custom_voice_descriptions: Optional[Dict[str, str]] = None,
                                 voice_samples: Optional[Dict[str, str]] = None) -> Dict[str, any]:
        """Complete enhanced pipeline processing"""
        self.stats['processing_start_time'] = time.time()
        logging.info(f"🚀 Starting Enhanced VoiceScore pipeline for: {url}")

        try:
            # Step 1: Enhanced web scraping
            raw_text = self.scrape_web_novel_enhanced(url)
            if not raw_text:
                raise ValueError("Enhanced scraping failed - no content extracted")

            # Step 2: Enhanced text processing with neural coreference
            sentences, characters, resolved_text, character_analysis = self.process_text_enhanced(raw_text)

            # Step 3: Create enhanced voice profiles with embeddings
            voice_profiles = self.create_enhanced_voice_profiles(
                characters, custom_voice_descriptions, voice_samples
            )

            # Step 4: Enhanced voice assignment with context awareness
            assignments = self.assign_voices_enhanced(sentences, characters, character_analysis)

            # Step 5: Enhanced audio synthesis
            audio_files = self.synthesize_audio_enhanced(assignments)

            # Step 6: Create enhanced outputs and reports
            outputs = self.create_enhanced_outputs(assignments, audio_files, character_analysis)

            # Save resolved text with metadata
            resolved_file = self.output_dir / "enhanced_resolved_text.txt"
            with open(resolved_file, 'w', encoding='utf-8') as f:
                f.write(f"Enhanced VoiceScore Processing Results\n")
                f.write(f"Original URL: {url}\n")
                f.write(f"Processed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Characters found: {len(characters)}\n")
                f.write("=" * 80 + "\n\n")
                f.write(resolved_text)

            # Final statistics
            self.stats['total_processing_time'] = time.time() - self.stats['processing_start_time']

            results = {
                'status': 'success',
                'output_directory': str(self.output_dir),
                'audio_files_count': len(audio_files),
                'characters_found': len(characters),
                'sentences_processed': len(sentences),
                'coreference_resolutions': self.stats['coreference_resolutions'],
                'total_processing_time': self.stats['total_processing_time'],
                'playlist_file': outputs['playlist'],
                'report_file': outputs['report'],
                'assignment_file': outputs['assignments'],
                'character_analysis': character_analysis,
                'embedding_stats': self.tts_engine.get_voice_analysis_report()['embedding_stats']
            }

            logging.info("🎉 Enhanced VoiceScore pipeline completed successfully!")
            return results

        except Exception as e:
            logging.error(f"❌ Enhanced pipeline failed: {e}")
            raise

def main():
    """Main function for enhanced VoiceScore pipeline"""
    # Enhanced logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("enhanced_voicescore.log"),
            logging.StreamHandler()
        ]
    )

    # Configuration
    url = "https://readnovelfull.com/absolute-resonance/chapter-1481-four-sword-realm-goddess-divineflame-mirror.html"

    # Enhanced voice descriptions with more detailed characteristics
    custom_voices = {
        "Li Luo": "confident, determined young male voice with slight heroic resonance and clear articulation",
        "Jiang Qing'e": "elegant, strong female voice with authoritative presence and refined tone",
        "Li Lingjing": "wise, mature female voice with caring undertones and measured speaking pace",
        "Yu Lang": "deep, powerful male voice with commanding authority and slight gravitas",
        "Song Yunfeng": "smooth, cultured male voice with scholarly precision and calm demeanor"
    }

    # Voice samples (if available)
    voice_samples = {
        # "Li Luo": "voice_samples/li_luo_sample.wav",
        # "Jiang Qing'e": "voice_samples/jiang_qinge_sample.wav"
    }

    # Initialize enhanced pipeline
    pipeline = EnhancedVoiceScorePipeline(
        output_dir="enhanced_novel_audio",
        device="auto",  # Will use CUDA if available
        use_neural_coref=True  # Enable neural coreference resolution
    )

    try:
        results = pipeline.process_web_novel_enhanced(
            url,
            custom_voice_descriptions=custom_voices,
            voice_samples=voice_samples
        )

        # Display enhanced results
        print("\n" + "=" * 80)
        print("🎉 Enhanced VoiceScore Processing Complete!")
        print("=" * 80)
        print(f"📁 Output directory: {results['output_directory']}")
        print(f"🎵 Audio files generated: {results['audio_files_count']}")
        print(f"👥 Characters identified: {results['characters_found']}")
        print(f"📝 Sentences processed: {results['sentences_processed']}")
        print(f"🔗 Coreference resolutions: {results['coreference_resolutions']}")
        print(f"⏱️  Total processing time: {results['total_processing_time']:.2f} seconds")
        print(f"📄 Detailed report: {results['report_file']}")
        print(f"🎼 Audio playlist: {results['playlist_file']}")

        # Embedding statistics
        embedding_stats = results['embedding_stats']
        print(f"\n🧠 Embedding Statistics:")
        print(f"   Speaker embeddings: {embedding_stats['speaker_embeddings']}")
        print(f"   Text-audio embeddings: {embedding_stats['text_audio_embeddings']}")
        print(f"   Text embeddings: {embedding_stats['text_embeddings']}")
        print("=" * 80)

    except Exception as e:
        logging.error(f"Enhanced pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()