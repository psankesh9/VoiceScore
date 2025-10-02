"""
Enhanced Multi-Speaker TTS System for VoiceScore
Integrates GlowTTS/VITS with ECAPA-TDNN speaker embeddings and CLAP text-to-audio matching
"""

import os
import logging
import numpy as np
import soundfile as sf
import torch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass

# TTS imports
from TTS.api import TTS
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer

# Enhanced voice embeddings
from enhanced_voice_embeddings import EnhancedVoiceEmbeddingManager, EnhancedVoiceProfile

# Voice analysis
import librosa

class EnhancedTTSEngine:
    """Enhanced TTS engine with ECAPA-TDNN speaker embeddings and CLAP matching"""

    def __init__(self, device: str = "auto", voice_sample_dir: str = "voice_samples"):
        self.device = self._setup_device(device)
        self.voice_manager = EnhancedVoiceEmbeddingManager()
        self.tts_models = {}
        self.current_model = None
        self.audio_cache = {}
        self.voice_sample_dir = Path(voice_sample_dir)
        self.voice_sample_dir.mkdir(exist_ok=True)

        # Speaker-to-TTS mapping for multi-speaker models
        self.speaker_mappings = {}

        # Initialize TTS models
        self._initialize_tts_models()

        # Initialize speaker database
        self._initialize_speaker_database()

    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _initialize_tts_models(self):
        """Initialize enhanced TTS models with multi-speaker support"""
        try:
            # Multi-speaker English models (ordered by quality)
            models_to_try = [
                ("tts_models/en/vctk/vits", "vctk_vits"),
                ("tts_models/en/ljspeech/glow-tts", "ljspeech_glow"),
                ("tts_models/en/ljspeech/tacotron2-DCA", "ljspeech_tacotron2")
            ]

            for model_name, model_key in models_to_try:
                try:
                    logging.info(f"Loading TTS model: {model_name}")
                    model = TTS(model_name).to(self.device)
                    self.tts_models[model_key] = model

                    # Set as current model if it's the first successful load
                    if self.current_model is None:
                        self.current_model = model
                        self.current_model_key = model_key

                        # Get available speakers for multi-speaker models
                        if hasattr(model, 'speakers') and model.speakers:
                            logging.info(f"✓ Multi-speaker model loaded with {len(model.speakers)} speakers")
                            self._analyze_tts_speakers(model.speakers, model_key)
                        else:
                            logging.info("✓ Single-speaker model loaded")

                    break  # Use first successful model

                except Exception as e:
                    logging.warning(f"Failed to load {model_name}: {e}")
                    continue

            if self.current_model is None:
                raise ValueError("Failed to load any TTS model")

            logging.info(f"TTS engine initialized with {self.current_model_key} on {self.device}")

        except Exception as e:
            logging.error(f"Failed to initialize TTS models: {e}")
            raise

    def _analyze_tts_speakers(self, speakers: List[str], model_key: str):
        """Analyze available TTS speakers and create voice characteristics"""
        speaker_info = {}

        # Known speaker characteristics for VCTK dataset
        vctk_speaker_info = {
            'p225': {'gender': 'female', 'age': 'adult', 'accent': 'english'},
            'p226': {'gender': 'male', 'age': 'adult', 'accent': 'english'},
            'p227': {'gender': 'male', 'age': 'adult', 'accent': 'english'},
            'p228': {'gender': 'female', 'age': 'adult', 'accent': 'english'},
            'p229': {'gender': 'female', 'age': 'adult', 'accent': 'english'},
            'p230': {'gender': 'female', 'age': 'adult', 'accent': 'scottish'},
            'p231': {'gender': 'female', 'age': 'adult', 'accent': 'english'},
            'p232': {'gender': 'male', 'age': 'adult', 'accent': 'english'},
            'p233': {'gender': 'female', 'age': 'adult', 'accent': 'english'},
            'p234': {'gender': 'female', 'age': 'adult', 'accent': 'scottish'},
            'p235': {'gender': 'female', 'age': 'adult', 'accent': 'english'},
            'p236': {'gender': 'female', 'age': 'adult', 'accent': 'english'},
            'p237': {'gender': 'male', 'age': 'adult', 'accent': 'english'},
            'p238': {'gender': 'female', 'age': 'adult', 'accent': 'english'},
            'p239': {'gender': 'female', 'age': 'adult', 'accent': 'english'},
            'p240': {'gender': 'female', 'age': 'adult', 'accent': 'irish'},
        }

        for speaker in speakers:
            if speaker in vctk_speaker_info:
                speaker_info[speaker] = vctk_speaker_info[speaker]
            else:
                # Default characteristics for unknown speakers
                speaker_info[speaker] = {
                    'gender': 'neutral',
                    'age': 'adult',
                    'accent': 'neutral'
                }

        self.speaker_mappings[model_key] = speaker_info
        logging.info(f"Analyzed {len(speaker_info)} TTS speakers for {model_key}")

    def _initialize_speaker_database(self):
        """Initialize speaker database with available TTS voices"""
        if not hasattr(self.current_model, 'speakers') or not self.current_model.speakers:
            logging.warning("Current TTS model doesn't support multiple speakers")
            return

        speakers = self.current_model.speakers
        speaker_info = self.speaker_mappings.get(self.current_model_key, {})

        # Create voice profiles for each TTS speaker
        for i, speaker_id in enumerate(speakers):
            info = speaker_info.get(speaker_id, {'gender': 'neutral', 'age': 'adult', 'accent': 'neutral'})

            # Create voice description
            description = f"{info['gender']} {info['age']} voice"
            if info['accent'] != 'neutral':
                description += f" with {info['accent']} accent"

            # Create enhanced profile for TTS speaker
            profile = self.voice_manager.create_voice_profile(
                character_name=f"TTS_Speaker_{speaker_id}",
                voice_id=1000 + i,  # High IDs for TTS speakers
                voice_description=description,
                gender=info['gender'],
                age_group=info['age'],
                accent=info['accent'],
                tts_speaker_id=speaker_id
            )

        logging.info(f"Initialized speaker database with {len(speakers)} TTS voices")

    def create_character_voice_profiles(self, characters: Dict[str, int],
                                      voice_descriptions: Optional[Dict[str, str]] = None,
                                      voice_samples: Optional[Dict[str, str]] = None) -> Dict[str, EnhancedVoiceProfile]:
        """Create enhanced voice profiles for characters"""
        profiles = {}

        for char_name, voice_id in characters.items():
            # Get voice description
            description = ""
            if voice_descriptions and char_name in voice_descriptions:
                description = voice_descriptions[char_name]
            else:
                description = self._generate_default_voice_description(char_name, voice_id)

            # Get voice sample path (if available)
            sample_path = None
            if voice_samples and char_name in voice_samples:
                sample_path = voice_samples[char_name]
                if not os.path.exists(sample_path):
                    logging.warning(f"Voice sample not found: {sample_path}")
                    sample_path = None

            # Extract additional attributes
            gender = self._infer_gender(char_name, description)
            age_group = self._infer_age_group(description)
            accent = self._infer_accent(description)

            # Create enhanced profile
            profile = self.voice_manager.create_voice_profile(
                character_name=char_name,
                voice_id=voice_id,
                voice_description=description,
                audio_sample_path=sample_path,
                gender=gender,
                age_group=age_group,
                accent=accent
            )

            profiles[char_name] = profile
            logging.info(f"Created enhanced voice profile for {char_name}: {description}")

        return profiles

    def _generate_default_voice_description(self, char_name: str, voice_id: int) -> str:
        """Generate enhanced default voice description"""
        voice_types = [
            "warm, clear, friendly voice",
            "deep, authoritative, confident voice",
            "soft, gentle, melodic voice",
            "energetic, youthful, bright voice",
            "calm, wise, measured voice",
            "strong, determined voice",
            "quiet, mysterious, thoughtful voice",
            "cheerful, upbeat, expressive voice",
            "sophisticated, elegant voice",
            "rough, gruff, weathered voice"
        ]

        base_description = voice_types[voice_id % len(voice_types)]

        # Enhanced character-specific traits
        name_lower = char_name.lower()

        if any(title in name_lower for title in ['king', 'emperor', 'lord', 'duke', 'sir', 'general']):
            return f"commanding, regal, {base_description} with noble bearing"
        elif any(title in name_lower for title in ['queen', 'empress', 'lady', 'duchess', 'princess']):
            return f"elegant, refined, {base_description} with aristocratic grace"
        elif any(title in name_lower for title in ['dr.', 'doctor', 'professor', 'scholar']):
            return f"intellectual, articulate, {base_description} with scholarly precision"
        elif any(title in name_lower for title in ['captain', 'major', 'colonel', 'lieutenant']):
            return f"disciplined, authoritative, {base_description} with military bearing"
        else:
            return base_description

    def _infer_gender(self, char_name: str, description: str) -> str:
        """Enhanced gender inference"""
        male_indicators = ['mr.', 'sir', 'king', 'lord', 'duke', 'prince', 'emperor', 'captain', 'major', 'colonel', 'general', 'he', 'his', 'him', 'male', 'man', 'boy', 'father', 'son', 'brother']
        female_indicators = ['ms.', 'mrs.', 'lady', 'queen', 'duchess', 'princess', 'empress', 'she', 'her', 'hers', 'female', 'woman', 'girl', 'mother', 'daughter', 'sister']

        text = f"{char_name.lower()} {description.lower()}"

        male_score = sum(2 if indicator in char_name.lower() else 1 for indicator in male_indicators if indicator in text)
        female_score = sum(2 if indicator in char_name.lower() else 1 for indicator in female_indicators if indicator in text)

        if male_score > female_score + 1:
            return "male"
        elif female_score > male_score + 1:
            return "female"
        else:
            return "neutral"

    def _infer_age_group(self, description: str) -> str:
        """Enhanced age group inference"""
        desc_lower = description.lower()

        if any(word in desc_lower for word in ['young', 'youthful', 'teenage', 'child', 'kid', 'boy', 'girl', 'adolescent']):
            return "young"
        elif any(word in desc_lower for word in ['old', 'elderly', 'aged', 'senior', 'ancient', 'wise', 'weathered', 'gruff']):
            return "elderly"
        elif any(word in desc_lower for word in ['middle-aged', 'mature', 'experienced']):
            return "middle_aged"
        else:
            return "adult"

    def _infer_accent(self, description: str) -> str:
        """Infer accent from description"""
        desc_lower = description.lower()

        accent_keywords = {
            'british': ['british', 'english', 'london', 'posh', 'aristocratic', 'noble'],
            'scottish': ['scottish', 'scots', 'highland'],
            'irish': ['irish', 'celtic'],
            'american': ['american', 'western', 'southern'],
            'neutral': ['neutral', 'standard', 'clear']
        }

        for accent, keywords in accent_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                return accent

        return "neutral"

    def select_best_tts_voice(self, character_name: str) -> str:
        """Select best TTS voice using enhanced embeddings"""
        if character_name not in self.voice_manager.voice_profiles:
            return self._get_default_speaker()

        profile = self.voice_manager.voice_profiles[character_name]

        # Use voice similarity search to find best TTS speaker
        similar_voices = self.voice_manager.find_similar_voices(
            query_description=profile.voice_description,
            query_audio_path=profile.audio_sample_path,
            k=3
        )

        # Find TTS speakers among similar voices
        for voice_name, similarity, match_type in similar_voices:
            if voice_name.startswith("TTS_Speaker_"):
                tts_speaker_id = voice_name.replace("TTS_Speaker_", "")
                if hasattr(self.current_model, 'speakers') and tts_speaker_id in self.current_model.speakers:
                    logging.info(f"Selected TTS speaker {tts_speaker_id} for {character_name} (similarity: {similarity:.3f}, type: {match_type})")
                    return tts_speaker_id

        # Fallback: use gender and age matching
        return self._select_speaker_by_attributes(profile)

    def _select_speaker_by_attributes(self, profile: EnhancedVoiceProfile) -> str:
        """Fallback speaker selection based on attributes"""
        if not hasattr(self.current_model, 'speakers') or not self.current_model.speakers:
            return self._get_default_speaker()

        speaker_info = self.speaker_mappings.get(self.current_model_key, {})

        # Find speakers matching gender
        matching_speakers = []
        for speaker, info in speaker_info.items():
            if info['gender'] == profile.gender or profile.gender == 'neutral':
                matching_speakers.append(speaker)

        if matching_speakers:
            # Select based on voice_id for consistency
            selected = matching_speakers[profile.voice_id % len(matching_speakers)]
            logging.info(f"Selected TTS speaker {selected} for {profile.character_name} by attributes")
            return selected

        return self._get_default_speaker()

    def _get_default_speaker(self) -> str:
        """Get default speaker"""
        if hasattr(self.current_model, 'speakers') and self.current_model.speakers:
            return self.current_model.speakers[0]
        return None

    def synthesize_speech(self, text: str, character_name: str, output_path: str) -> str:
        """Synthesize speech with enhanced voice selection"""
        try:
            # Handle empty or very short text
            if not text.strip() or len(text.strip()) < 3:
                self._generate_silence(output_path)
                return output_path

            # Select best TTS voice
            speaker_id = self.select_best_tts_voice(character_name)

            # Generate speech
            if speaker_id and hasattr(self.current_model, 'speakers'):
                # Multi-speaker synthesis
                self.current_model.tts_to_file(
                    text=text,
                    speaker=speaker_id,
                    file_path=output_path
                )
                logging.info(f"Generated speech for {character_name} using speaker {speaker_id}: {text[:50]}...")
            else:
                # Single-speaker synthesis
                self.current_model.tts_to_file(
                    text=text,
                    file_path=output_path
                )
                logging.info(f"Generated speech for {character_name} (single-speaker): {text[:50]}...")

            return output_path

        except Exception as e:
            logging.error(f"Enhanced TTS synthesis error for {character_name}: {e}")
            self._generate_silence(output_path)
            return output_path

    def _generate_silence(self, output_path: str, duration: float = 0.5):
        """Generate silence for failed synthesis"""
        silence = np.zeros(int(22050 * duration))
        sf.write(output_path, silence, 22050)

    def get_voice_analysis_report(self) -> Dict:
        """Generate comprehensive voice analysis report"""
        report = {
            'tts_engine': {
                'current_model': self.current_model_key,
                'device': self.device,
                'available_speakers': len(getattr(self.current_model, 'speakers', [])),
                'speaker_list': getattr(self.current_model, 'speakers', [])
            },
            'embedding_stats': self.voice_manager.get_embedding_stats(),
            'character_profiles': {}
        }

        # Add character profile details
        for name, profile in self.voice_manager.voice_profiles.items():
            if not name.startswith("TTS_Speaker_"):
                report['character_profiles'][name] = {
                    'voice_id': profile.voice_id,
                    'description': profile.voice_description,
                    'gender': profile.gender,
                    'age_group': profile.age_group,
                    'accent': profile.accent,
                    'has_audio_sample': profile.audio_sample_path is not None,
                    'embedding_types': {
                        'speaker': profile.speaker_embedding is not None,
                        'text_audio': profile.text_audio_embedding is not None,
                        'text': profile.text_embedding is not None
                    }
                }

        return report

    def save_voice_profiles(self, filepath: str):
        """Save enhanced voice profiles"""
        profiles_data = {}

        for name, profile in self.voice_manager.voice_profiles.items():
            if not name.startswith("TTS_Speaker_"):  # Don't save TTS speaker profiles
                profiles_data[name] = {
                    'voice_id': profile.voice_id,
                    'description': profile.voice_description,
                    'gender': profile.gender,
                    'age_group': profile.age_group,
                    'accent': profile.accent,
                    'audio_sample_path': profile.audio_sample_path,
                    'tts_speaker_id': profile.tts_speaker_id
                }

        with open(filepath, 'w') as f:
            json.dump(profiles_data, f, indent=2)

        logging.info(f"Saved {len(profiles_data)} enhanced voice profiles to {filepath}")

def main():
    """Test the enhanced TTS system"""
    logging.basicConfig(level=logging.INFO)

    # Initialize enhanced TTS engine
    tts_engine = EnhancedTTSEngine()

    # Test characters with enhanced descriptions
    characters = {
        "Sarah": 2,
        "John": 3,
        "Dr. Smith": 4
    }

    # Enhanced voice descriptions
    voice_descriptions = {
        "Sarah": "young, cheerful, bright female voice with warm, friendly tone",
        "John": "confident, deep male voice with authoritative presence and slight baritone quality",
        "Dr. Smith": "intellectual, precise, elderly male voice with scholarly articulation and measured pace"
    }

    # Create enhanced profiles
    profiles = tts_engine.create_character_voice_profiles(characters, voice_descriptions)

    # Test synthesis
    test_segments = [
        ("Hello everyone, I'm excited to be here today!", "Sarah"),
        ("Good morning. Let's get straight to business.", "John"),
        ("The experimental results demonstrate a fascinating correlation.", "Dr. Smith")
    ]

    output_files = []
    for i, (text, character) in enumerate(test_segments):
        output_file = f"enhanced_test_{i}_{character}.wav"
        tts_engine.synthesize_speech(text, character, output_file)
        output_files.append(output_file)

    print(f"Generated {len(output_files)} enhanced audio files")

    # Generate analysis report
    report = tts_engine.get_voice_analysis_report()
    print("\nVoice Analysis Report:")
    print(json.dumps(report, indent=2))

    # Save profiles
    tts_engine.save_voice_profiles("enhanced_voice_profiles.json")

if __name__ == "__main__":
    main()