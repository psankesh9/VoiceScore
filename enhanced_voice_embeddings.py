"""
Enhanced Voice Embedding Manager with ECAPA-TDNN and CLAP support
Better voice similarity matching using actual audio embeddings
"""

import os
import logging
import numpy as np
import torch
import torchaudio
import faiss
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

# Audio processing
import librosa
import soundfile as sf

# Speaker embeddings
try:
    import speechbrain as sb
    from speechbrain.pretrained import EncoderClassifier
    HAS_SPEECHBRAIN = True
except ImportError:
    logging.warning("SpeechBrain not available. Install with: pip install speechbrain")
    HAS_SPEECHBRAIN = False

# Multimodal embeddings
try:
    import laion_clap
    HAS_CLAP = True
except ImportError:
    logging.warning("CLAP not available. Install with: pip install laion-clap")
    HAS_CLAP = False

# Fallback to sentence transformers
from sentence_transformers import SentenceTransformer

@dataclass
class EnhancedVoiceProfile:
    """Enhanced voice profile with multiple embedding types"""
    character_name: str
    voice_id: int

    # Embeddings
    speaker_embedding: Optional[np.ndarray] = None  # From audio (ECAPA-TDNN)
    text_audio_embedding: Optional[np.ndarray] = None  # From description (CLAP)
    text_embedding: Optional[np.ndarray] = None  # From description (SentenceTransformer)

    # Voice characteristics
    voice_description: str = ""
    gender: str = "neutral"
    age_group: str = "adult"
    accent: str = "neutral"

    # Audio properties (if available)
    fundamental_frequency: Optional[float] = None
    speaking_rate: Optional[float] = None
    audio_sample_path: Optional[str] = None

    # TTS properties
    tts_speaker_id: Optional[str] = None
    voice_quality_score: float = 0.5

class EnhancedVoiceEmbeddingManager:
    """Enhanced voice embedding manager supporting multiple embedding types"""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.voice_profiles = {}

        # Multiple FAISS indices for different embedding types
        self.speaker_index = faiss.IndexFlatL2(embedding_dim) if embedding_dim else None
        self.text_audio_index = faiss.IndexFlatL2(embedding_dim) if embedding_dim else None
        self.text_index = faiss.IndexFlatL2(embedding_dim) if embedding_dim else None

        self.profile_ids = []  # Maps FAISS indices to character names

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize embedding models"""
        self.models = {}

        # 1. Speaker embedding model (ECAPA-TDNN)
        if HAS_SPEECHBRAIN:
            try:
                logging.info("Loading ECAPA-TDNN speaker embedding model...")
                self.models['speaker'] = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                logging.info("✓ ECAPA-TDNN loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load ECAPA-TDNN: {e}")
                self.models['speaker'] = None
        else:
            self.models['speaker'] = None

        # 2. Text-to-Audio model (CLAP)
        if HAS_CLAP:
            try:
                logging.info("Loading CLAP multimodal model...")
                self.models['clap'] = laion_clap.CLAP_Module(enable_fusion=False)
                self.models['clap'].load_ckpt()  # Load pretrained weights
                logging.info("✓ CLAP loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load CLAP: {e}")
                self.models['clap'] = None
        else:
            self.models['clap'] = None

        # 3. Fallback text embedding model
        try:
            logging.info("Loading sentence transformer (fallback)...")
            self.models['text'] = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("✓ SentenceTransformer loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer: {e}")
            self.models['text'] = None

    def extract_speaker_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio file using ECAPA-TDNN"""
        if not self.models.get('speaker'):
            return None

        try:
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Extract embedding
            with torch.no_grad():
                embeddings = self.models['speaker'].encode_batch(waveform)
                embedding = embeddings.squeeze().cpu().numpy()

            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)

            # Pad/truncate to match embedding dimension
            if len(embedding) < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])
            else:
                embedding = embedding[:self.embedding_dim]

            return embedding.astype('float32')

        except Exception as e:
            logging.error(f"Failed to extract speaker embedding from {audio_path}: {e}")
            return None

    def extract_text_audio_embedding(self, text_description: str) -> Optional[np.ndarray]:
        """Extract text-to-audio embedding using CLAP"""
        if not self.models.get('clap'):
            return None

        try:
            # CLAP expects text in specific format
            text_data = [text_description]

            with torch.no_grad():
                text_embed = self.models['clap'].get_text_embedding(text_data)
                embedding = text_embed.squeeze().cpu().numpy()

            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)

            # Pad/truncate to match embedding dimension
            if len(embedding) < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])
            else:
                embedding = embedding[:self.embedding_dim]

            return embedding.astype('float32')

        except Exception as e:
            logging.error(f"Failed to extract CLAP embedding: {e}")
            return None

    def extract_text_embedding(self, text_description: str) -> np.ndarray:
        """Extract text embedding using SentenceTransformer (fallback)"""
        if not self.models.get('text'):
            # Return random embedding as last resort
            return np.random.randn(self.embedding_dim).astype('float32')

        try:
            embedding = self.models['text'].encode(text_description)

            # Pad/truncate to match embedding dimension
            if len(embedding) < self.embedding_dim:
                padding = np.zeros(self.embedding_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])
            else:
                embedding = embedding[:self.embedding_dim]

            return embedding.astype('float32')

        except Exception as e:
            logging.error(f"Failed to extract text embedding: {e}")
            return np.random.randn(self.embedding_dim).astype('float32')

    def create_voice_profile(self, character_name: str, voice_id: int,
                           voice_description: str = "",
                           audio_sample_path: Optional[str] = None,
                           **kwargs) -> EnhancedVoiceProfile:
        """Create enhanced voice profile with multiple embedding types"""

        profile = EnhancedVoiceProfile(
            character_name=character_name,
            voice_id=voice_id,
            voice_description=voice_description,
            audio_sample_path=audio_sample_path,
            **kwargs
        )

        # Extract different types of embeddings

        # 1. Speaker embedding from audio (highest priority)
        if audio_sample_path and os.path.exists(audio_sample_path):
            profile.speaker_embedding = self.extract_speaker_embedding(audio_sample_path)
            if profile.speaker_embedding is not None:
                logging.info(f"✓ Extracted speaker embedding for {character_name}")

        # 2. Text-to-audio embedding (medium priority)
        if voice_description:
            profile.text_audio_embedding = self.extract_text_audio_embedding(voice_description)
            if profile.text_audio_embedding is not None:
                logging.info(f"✓ Extracted CLAP embedding for {character_name}")

        # 3. Text embedding (fallback)
        if voice_description:
            profile.text_embedding = self.extract_text_embedding(voice_description)
            logging.info(f"✓ Extracted text embedding for {character_name}")

        # Add to appropriate indices
        self._add_profile_to_indices(profile)

        # Store profile
        self.voice_profiles[character_name] = profile
        self.profile_ids.append(character_name)

        return profile

    def _add_profile_to_indices(self, profile: EnhancedVoiceProfile):
        """Add voice profile to appropriate FAISS indices"""

        # Add to speaker index (highest quality)
        if profile.speaker_embedding is not None and self.speaker_index is not None:
            embedding = profile.speaker_embedding.reshape(1, -1)
            self.speaker_index.add(embedding)

        # Add to text-audio index (medium quality)
        elif profile.text_audio_embedding is not None and self.text_audio_index is not None:
            embedding = profile.text_audio_embedding.reshape(1, -1)
            self.text_audio_index.add(embedding)

        # Add to text index (fallback)
        elif profile.text_embedding is not None and self.text_index is not None:
            embedding = profile.text_embedding.reshape(1, -1)
            self.text_index.add(embedding)

    def find_similar_voices(self, query_description: str = "",
                          query_audio_path: str = "",
                          k: int = 3) -> List[Tuple[str, float, str]]:
        """Find similar voices using best available embedding type"""

        results = []

        # Priority 1: Audio-based search (if audio query provided)
        if query_audio_path and os.path.exists(query_audio_path):
            query_embedding = self.extract_speaker_embedding(query_audio_path)
            if query_embedding is not None and self.speaker_index.ntotal > 0:
                distances, indices = self.speaker_index.search(
                    query_embedding.reshape(1, -1), min(k, self.speaker_index.ntotal)
                )
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < len(self.profile_ids):
                        similarity = 1.0 / (1.0 + dist)
                        results.append((self.profile_ids[idx], similarity, "speaker_audio"))

        # Priority 2: Text-to-audio search (if description provided and no audio results)
        elif query_description and not results:
            query_embedding = self.extract_text_audio_embedding(query_description)
            if query_embedding is not None and self.text_audio_index.ntotal > 0:
                distances, indices = self.text_audio_index.search(
                    query_embedding.reshape(1, -1), min(k, self.text_audio_index.ntotal)
                )
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < len(self.profile_ids):
                        similarity = 1.0 / (1.0 + dist)
                        results.append((self.profile_ids[idx], similarity, "text_audio"))

        # Priority 3: Text-only search (fallback)
        if query_description and not results:
            query_embedding = self.extract_text_embedding(query_description)
            if self.text_index.ntotal > 0:
                distances, indices = self.text_index.search(
                    query_embedding.reshape(1, -1), min(k, self.text_index.ntotal)
                )
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < len(self.profile_ids):
                        similarity = 1.0 / (1.0 + dist)
                        results.append((self.profile_ids[idx], similarity, "text_only"))

        return results[:k]

    def get_embedding_stats(self) -> Dict[str, int]:
        """Get statistics about available embeddings"""
        return {
            'speaker_embeddings': self.speaker_index.ntotal if self.speaker_index else 0,
            'text_audio_embeddings': self.text_audio_index.ntotal if self.text_audio_index else 0,
            'text_embeddings': self.text_index.ntotal if self.text_index else 0,
            'total_profiles': len(self.voice_profiles)
        }

def main():
    """Test the enhanced voice embedding system"""
    logging.basicConfig(level=logging.INFO)

    # Initialize enhanced manager
    manager = EnhancedVoiceEmbeddingManager()

    # Test with text descriptions
    manager.create_voice_profile(
        character_name="Sarah",
        voice_id=2,
        voice_description="young, cheerful, bright female voice"
    )

    manager.create_voice_profile(
        character_name="John",
        voice_id=3,
        voice_description="confident, deep male voice"
    )

    # Show embedding statistics
    stats = manager.get_embedding_stats()
    print("Embedding Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test similarity search
    similar = manager.find_similar_voices("warm, friendly female voice", k=2)
    print(f"\nSimilar to 'warm, friendly female voice': {similar}")

if __name__ == "__main__":
    main()