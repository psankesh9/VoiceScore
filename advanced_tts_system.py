"""
Advanced Multi-Speaker TTS System for VoiceScore
Integrates GlowTTS/VITS with character embeddings and voice cloning
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
from sentence_transformers import SentenceTransformer

# TTS imports
from TTS.api import TTS
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer

# !pip install --upgrade numba

# Voice analysis
import librosa
import faiss

@dataclass
class VoiceProfile:
    """Represents a character's voice profile"""
    character_name: str
    voice_id: int
    embedding: np.ndarray
    voice_description: str = ""
    gender: str = "neutral"
    age_group: str = "adult"
    accent: str = "neutral"
    pitch_range: Tuple[float, float] = (80.0, 300.0)
    speaking_rate: float = 1.0
    tts_model_path: Optional[str] = None
    sample_audio_path: Optional[str] = None

class VoiceEmbeddingManager:
    """Manages voice embeddings and similarity matching"""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.voice_profiles = {}
        self.embedding_index = faiss.IndexFlatL2(embedding_dim)
        self.profile_ids = []  # Maps FAISS index to character names
        self.sentence_embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def add_voice_profile(self, profile: VoiceProfile):
        """Add a voice profile to the database"""
        self.voice_profiles[profile.character_name] = profile

        # Add to FAISS index for similarity search
        embedding = profile.embedding.reshape(1, -1).astype('float32')
        self.embedding_index.add(embedding)
        self.profile_ids.append(profile.character_name)

        logging.info(f"Added voice profile for {profile.character_name}")

    def find_similar_voices(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Find voices similar to the query embedding"""
        query = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.embedding_index.search(query, k)

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.profile_ids):
                char_name = self.profile_ids[idx]
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                results.append((char_name, similarity))

        return results

    def generate_voice_embedding_from_description(self, description: str) -> np.ndarray:
        """Generate voice embedding from textual description"""
        # Use sentence transformer to encode voice description
        text_embedding = self.sentence_embedder.encode(description)

        # Pad or truncate to match embedding dimension
        if len(text_embedding) < self.embedding_dim:
            padding = np.zeros(self.embedding_dim - len(text_embedding))
            voice_embedding = np.concatenate([text_embedding, padding])
        else:
            voice_embedding = text_embedding[:self.embedding_dim]

        return voice_embedding.astype('float32')

    def create_voice_profile_from_description(self, character_name: str, voice_id: int,
                                           description: str, **kwargs) -> VoiceProfile:
        """Create a voice profile from textual description"""
        embedding = self.generate_voice_embedding_from_description(description)

        profile = VoiceProfile(
            character_name=character_name,
            voice_id=voice_id,
            embedding=embedding,
            voice_description=description,
            **kwargs
        )

        self.add_voice_profile(profile)
        return profile

class AdvancedTTSEngine:
    """Advanced TTS engine with multi-speaker synthesis"""

    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.voice_manager = VoiceEmbeddingManager()
        self.tts_models = {}
        self.current_model = None
        self.audio_cache = {}

        # Initialize default TTS models
        self._initialize_default_models()

    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _initialize_default_models(self):
        """Initialize default TTS models"""
        try:
            # Multi-speaker English model
            logging.info("Loading multi-speaker TTS model...")
            self.tts_models['multi_speaker'] = TTS("tts_models/en/vctk/vits").to(self.device)

            # Single speaker model for voice cloning
            logging.info("Loading voice cloning model...")
            self.tts_models['voice_clone'] = TTS("tts_models/en/ljspeech/glow-tts").to(self.device)

            self.current_model = self.tts_models['multi_speaker']
            logging.info(f"TTS models initialized on {self.device}")

        except Exception as e:
            logging.error(f"Failed to initialize TTS models: {e}")
            # Fallback to basic model
            self.tts_models['basic'] = TTS("tts_models/en/ljspeech/tacotron2-DCA").to(self.device)
            self.current_model = self.tts_models['basic']

    def create_character_voice_profiles(self, characters: Dict[str, int],
                                      voice_descriptions: Optional[Dict[str, str]] = None) -> Dict[str, VoiceProfile]:
        """Create voice profiles for characters"""
        profiles = {}

        for char_name, voice_id in characters.items():
            # Get voice description or use defaults
            if voice_descriptions and char_name in voice_descriptions:
                description = voice_descriptions[char_name]
            else:
                description = self._generate_default_voice_description(char_name, voice_id)

            # Extract gender and other attributes from description or name
            gender = self._infer_gender(char_name, description)
            age_group = self._infer_age_group(description)

            profile = self.voice_manager.create_voice_profile_from_description(
                character_name=char_name,
                voice_id=voice_id,
                description=description,
                gender=gender,
                age_group=age_group
            )

            profiles[char_name] = profile
            logging.info(f"Created voice profile for {char_name}: {description}")

        return profiles

    def _generate_default_voice_description(self, char_name: str, voice_id: int) -> str:
        """Generate default voice description based on character name and ID"""
        # Simple heuristics for voice assignment
        voice_types = [
            "clear, warm, friendly voice",
            "deep, authoritative voice",
            "soft, gentle, melodic voice",
            "energetic, youthful voice",
            "calm, wise, measured voice",
            "confident, strong voice",
            "quiet, mysterious voice",
            "cheerful, bright voice"
        ]

        base_description = voice_types[voice_id % len(voice_types)]

        # Add character-specific traits
        if any(title in char_name.lower() for title in ['king', 'lord', 'duke', 'sir']):
            return f"regal, commanding, {base_description}"
        elif any(title in char_name.lower() for title in ['queen', 'lady', 'duchess']):
            return f"elegant, refined, {base_description}"
        elif 'dr.' in char_name.lower() or 'professor' in char_name.lower():
            return f"intellectual, precise, {base_description}"
        else:
            return base_description

    def _infer_gender(self, char_name: str, description: str) -> str:
        """Infer gender from character name and description"""
        male_indicators = ['mr.', 'sir', 'king', 'lord', 'duke', 'prince', 'he', 'his', 'him']
        female_indicators = ['ms.', 'mrs.', 'lady', 'queen', 'duchess', 'princess', 'she', 'her']

        name_lower = char_name.lower()
        desc_lower = description.lower()
        text = f"{name_lower} {desc_lower}"

        male_count = sum(1 for indicator in male_indicators if indicator in text)
        female_count = sum(1 for indicator in female_indicators if indicator in text)

        if male_count > female_count:
            return "male"
        elif female_count > male_count:
            return "female"
        else:
            return "neutral"

    def _infer_age_group(self, description: str) -> str:
        """Infer age group from description"""
        if any(word in description.lower() for word in ['young', 'youthful', 'child', 'teenage']):
            return "young"
        elif any(word in description.lower() for word in ['old', 'elderly', 'wise', 'aged']):
            return "elderly"
        else:
            return "adult"

    def select_tts_voice(self, character_name: str) -> str:
        """Select appropriate TTS voice for character"""
        if character_name not in self.voice_manager.voice_profiles:
            return "p225"  # Default voice

        profile = self.voice_manager.voice_profiles[character_name]

        # Map character attributes to available voices
        # This is a simplified mapping - in practice, you'd have more sophisticated selection
        if hasattr(self.current_model, 'speakers') and self.current_model.speakers:
            available_speakers = self.current_model.speakers

            if profile.gender == "male":
                male_voices = [s for s in available_speakers if s.startswith(('p2', 'p3')) and s not in ['p225', '229', '230']]
                return male_voices[0] if male_voices else available_speakers[0]
            elif profile.gender == "female":
                female_voices = [s for s in available_speakers if s.startswith(('p2', 'p1')) or s in ['p225', '229', '230']]
                return female_voices[0] if female_voices else available_speakers[1]
            else:
                return available_speakers[profile.voice_id % len(available_speakers)]

        return "p225"  # Fallback

    def synthesize_speech(self, text: str, character_name: str, output_path: str) -> str:
        """Synthesize speech for a specific character"""
        try:
            # Handle empty or very short text
            if not text.strip() or len(text.strip()) < 3:
                self._generate_silence(output_path)
                return output_path

            # Get character voice profile
            speaker_id = self.select_tts_voice(character_name)

            # Generate speech
            if 'multi_speaker' in self.tts_models and hasattr(self.current_model, 'speakers'):
                # Multi-speaker model
                self.current_model.tts_to_file(
                    text=text,
                    speaker=speaker_id,
                    file_path=output_path
                )
            else:
                # Single speaker model
                self.current_model.tts_to_file(
                    text=text,
                    file_path=output_path
                )

            logging.info(f"Generated speech for {character_name}: {text[:50]}...")
            return output_path

        except Exception as e:
            logging.error(f"TTS synthesis error for {character_name}: {e}")
            self._generate_silence(output_path)
            return output_path

    def _generate_silence(self, output_path: str, duration: float = 0.5):
        """Generate silence for failed synthesis"""
        silence = np.zeros(int(22050 * duration))
        sf.write(output_path, silence, 22050)

    def batch_synthesize(self, text_segments: List[Tuple[str, str]], output_dir: str) -> List[str]:
        """Batch synthesize multiple text segments"""
        output_files = []
        os.makedirs(output_dir, exist_ok=True)

        for i, (text, character) in enumerate(text_segments):
            output_file = os.path.join(output_dir, f"segment_{i:04d}_{character}.wav")
            self.synthesize_speech(text, character, output_file)
            output_files.append(output_file)

        return output_files

    def save_voice_profiles(self, filepath: str):
        """Save voice profiles to file"""
        profiles_data = {}
        for name, profile in self.voice_manager.voice_profiles.items():
            profiles_data[name] = {
                'voice_id': profile.voice_id,
                'description': profile.voice_description,
                'gender': profile.gender,
                'age_group': profile.age_group,
                'accent': profile.accent,
                'embedding': profile.embedding.tolist()
            }

        with open(filepath, 'w') as f:
            json.dump(profiles_data, f, indent=2)

        logging.info(f"Saved voice profiles to {filepath}")

    def load_voice_profiles(self, filepath: str):
        """Load voice profiles from file"""
        if not os.path.exists(filepath):
            logging.warning(f"Voice profiles file not found: {filepath}")
            return

        with open(filepath, 'r') as f:
            profiles_data = json.load(f)

        for name, data in profiles_data.items():
            embedding = np.array(data['embedding'], dtype=np.float32)

            profile = VoiceProfile(
                character_name=name,
                voice_id=data['voice_id'],
                embedding=embedding,
                voice_description=data['description'],
                gender=data['gender'],
                age_group=data['age_group'],
                accent=data.get('accent', 'neutral')
            )

            self.voice_manager.add_voice_profile(profile)

        logging.info(f"Loaded {len(profiles_data)} voice profiles from {filepath}")

def main():
    """Test the advanced TTS system"""
    logging.basicConfig(level=logging.INFO)

    # Initialize TTS engine
    tts_engine = AdvancedTTSEngine()

    # Test characters
    characters = {
        "Sarah": 2,
        "John": 3,
        "Dr. Smith": 4
    }

    # Create voice profiles
    voice_descriptions = {
        "Sarah": "young, cheerful, bright female voice",
        "John": "confident, deep male voice",
        "Dr. Smith": "intellectual, precise, elderly male voice"
    }

    profiles = tts_engine.create_character_voice_profiles(characters, voice_descriptions)

    # Test synthesis
    test_segments = [
        ("Hello, how are you today?", "Sarah"),
        ("I'm doing well, thank you for asking.", "John"),
        ("The experimental results are quite fascinating.", "Dr. Smith")
    ]

    output_files = tts_engine.batch_synthesize(test_segments, "test_output")
    print(f"Generated {len(output_files)} audio files")

    # Save profiles
    tts_engine.save_voice_profiles("voice_profiles.json")

if __name__ == "__main__":
    main()
