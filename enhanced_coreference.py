"""
Enhanced Coreference Resolution System for VoiceScore
Advanced character tracking with neural coreference resolution and context awareness
"""

import os
import json
import logging
import spacy
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
import re

# Enhanced NLP imports
try:
    import neuralcoref
    HAS_NEURALCOREF = True
except ImportError:
    logging.warning("neuralcoref not available. Install with: pip install neuralcoref")
    HAS_NEURALCOREF = False

try:
    from allennlp.predictors.predictor import Predictor
    HAS_ALLENNLP = True
except ImportError:
    logging.warning("allennlp not available. Install with: pip install allennlp allennlp-models")
    HAS_ALLENNLP = False

class EnhancedCharacterMemory:
    """Advanced character memory with neural embeddings and relationship tracking"""

    def __init__(self, max_context_window=15, embedding_model='all-MiniLM-L6-v2'):
        self.characters = {}
        self.context_window = deque(maxlen=max_context_window)
        self.sentence_embedder = SentenceTransformer(embedding_model)
        self.character_mentions = defaultdict(list)
        self.character_relationships = defaultdict(dict)
        self.emotional_states = defaultdict(str)
        self.conversation_history = defaultdict(list)

        # Enhanced tracking
        self.character_aliases = defaultdict(set)
        self.character_descriptions = defaultdict(list)
        self.scene_context = []

    def add_character(self, name: str, voice_id: int, context: str = "", aliases: List[str] = None):
        """Add character with enhanced information tracking"""
        if name not in self.characters:
            self.characters[name] = {
                'name': name,
                'voice_id': voice_id,
                'aliases': [name],
                'context': context,
                'embedding': self.sentence_embedder.encode(f"{name} {context}"),
                'last_mentioned': 0,
                'mention_count': 0,
                'emotional_state': 'neutral',
                'relationships': {},
                'speaking_patterns': [],
                'physical_description': "",
                'role': "character",  # character, narrator, etc.
                'importance_score': 0.0
            }

        # Add aliases
        if aliases:
            for alias in aliases:
                self.character_aliases[name].add(alias)
                self.characters[name]['aliases'].append(alias)

        self.characters[name]['mention_count'] += 1
        self.characters[name]['importance_score'] = self._calculate_importance(name)
        return self.characters[name]

    def _calculate_importance(self, character_name: str) -> float:
        """Calculate character importance based on mentions and context"""
        if character_name not in self.characters:
            return 0.0

        char_info = self.characters[character_name]
        mention_score = min(char_info['mention_count'] / 10.0, 1.0)
        recency_score = 1.0 - (len(self.context_window) - char_info['last_mentioned']) / len(self.context_window) if char_info['last_mentioned'] > 0 else 0.0
        relationship_score = len(char_info['relationships']) / max(len(self.characters) - 1, 1)

        return (mention_score * 0.4 + recency_score * 0.3 + relationship_score * 0.3)

    def add_relationship(self, char1: str, char2: str, relationship_type: str):
        """Track relationships between characters"""
        if char1 in self.characters and char2 in self.characters:
            self.characters[char1]['relationships'][char2] = relationship_type
            self.characters[char2]['relationships'][char1] = relationship_type
            self.character_relationships[char1][char2] = relationship_type

    def update_emotional_state(self, character_name: str, emotion: str, context: str):
        """Update character's emotional state with context"""
        if character_name in self.characters:
            self.characters[character_name]['emotional_state'] = emotion
            self.emotional_states[character_name] = emotion

    def get_character_by_pronoun(self, pronoun: str, sentence_idx: int, context: str,
                               sentence_embedding: np.ndarray = None) -> Optional[str]:
        """Enhanced pronoun resolution using multiple strategies"""

        candidates = self._get_pronoun_candidates(pronoun, sentence_idx)

        if not candidates:
            return None

        # Multiple resolution strategies
        scores = {}

        for char_name in candidates:
            score = 0.0

            # 1. Recency score (closer mentions = higher score)
            if self.character_mentions[char_name]:
                last_mention = max(self.character_mentions[char_name])
                recency_score = 1.0 / (sentence_idx - last_mention + 1)
                score += recency_score * 0.4

            # 2. Importance score
            score += self.characters[char_name]['importance_score'] * 0.3

            # 3. Context similarity (if embeddings available)
            if sentence_embedding is not None:
                char_embedding = self.characters[char_name]['embedding']
                similarity = np.dot(sentence_embedding, char_embedding) / (
                    np.linalg.norm(sentence_embedding) * np.linalg.norm(char_embedding)
                )
                score += similarity * 0.2

            # 4. Conversation continuity
            if self._in_recent_conversation(char_name, sentence_idx):
                score += 0.1

            scores[char_name] = score

        # Return highest scoring candidate
        if scores:
            best_candidate = max(scores.items(), key=lambda x: x[1])
            return best_candidate[0]

        return None

    def _get_pronoun_candidates(self, pronoun: str, sentence_idx: int) -> List[str]:
        """Get candidate characters for pronoun resolution"""
        pronoun_lower = pronoun.lower()

        # Gender-based filtering
        male_pronouns = {'he', 'him', 'his', 'himself'}
        female_pronouns = {'she', 'her', 'hers', 'herself'}
        neutral_pronouns = {'they', 'them', 'their', 'themselves'}

        candidates = []

        # Look for recently mentioned characters
        for char_name, char_info in self.characters.items():
            if not self.character_mentions[char_name]:
                continue

            last_mention = max(self.character_mentions[char_name])
            if sentence_idx - last_mention > 10:  # Too far back
                continue

            # Gender filtering
            char_gender = self._infer_character_gender(char_name, char_info)

            if pronoun_lower in male_pronouns and char_gender == 'male':
                candidates.append(char_name)
            elif pronoun_lower in female_pronouns and char_gender == 'female':
                candidates.append(char_name)
            elif pronoun_lower in neutral_pronouns:
                candidates.append(char_name)

        return candidates

    def _infer_character_gender(self, char_name: str, char_info: Dict) -> str:
        """Enhanced gender inference"""
        # Check context and aliases for gender indicators
        full_text = f"{char_name} {char_info['context']} {' '.join(char_info['aliases'])}"

        male_indicators = ['mr.', 'sir', 'king', 'lord', 'duke', 'prince', 'emperor', 'he', 'his', 'him', 'male', 'man', 'boy', 'father', 'son', 'brother', 'king', 'captain', 'general']
        female_indicators = ['ms.', 'mrs.', 'lady', 'queen', 'duchess', 'princess', 'empress', 'she', 'her', 'hers', 'female', 'woman', 'girl', 'mother', 'daughter', 'sister', 'queen']

        male_count = sum(1 for indicator in male_indicators if indicator in full_text.lower())
        female_count = sum(1 for indicator in female_indicators if indicator in full_text.lower())

        if male_count > female_count:
            return 'male'
        elif female_count > male_count:
            return 'female'
        else:
            return 'neutral'

    def _in_recent_conversation(self, char_name: str, sentence_idx: int, window: int = 5) -> bool:
        """Check if character was in recent conversation"""
        recent_mentions = [idx for idx in self.character_mentions[char_name]
                          if sentence_idx - window <= idx < sentence_idx]
        return len(recent_mentions) > 0

    def update_context(self, sentence: str, sentence_idx: int, characters_in_sentence: List[str],
                      sentence_embedding: np.ndarray = None):
        """Enhanced context updating with embeddings and relationships"""
        context_entry = {
            'sentence': sentence,
            'index': sentence_idx,
            'characters': characters_in_sentence,
            'embedding': sentence_embedding,
            'scene_info': self._extract_scene_info(sentence)
        }

        self.context_window.append(context_entry)

        # Update character mentions and relationships
        for char_name in characters_in_sentence:
            self.character_mentions[char_name].append(sentence_idx)
            if char_name in self.characters:
                self.characters[char_name]['last_mentioned'] = sentence_idx

        # Update character relationships based on co-occurrence
        if len(characters_in_sentence) > 1:
            for i, char1 in enumerate(characters_in_sentence):
                for char2 in characters_in_sentence[i+1:]:
                    if char1 in self.characters and char2 in self.characters:
                        # Increment interaction count
                        if char2 not in self.characters[char1]['relationships']:
                            self.characters[char1]['relationships'][char2] = 'interacts_with'
                        if char1 not in self.characters[char2]['relationships']:
                            self.characters[char2]['relationships'][char1] = 'interacts_with'

    def _extract_scene_info(self, sentence: str) -> Dict[str, any]:
        """Extract scene information from sentence"""
        scene_info = {
            'location_mentioned': False,
            'time_mentioned': False,
            'action_type': 'dialogue',  # dialogue, narrative, action
            'emotional_tone': 'neutral'
        }

        # Location indicators
        location_words = ['room', 'house', 'outside', 'inside', 'building', 'street', 'park', 'office']
        if any(word in sentence.lower() for word in location_words):
            scene_info['location_mentioned'] = True

        # Time indicators
        time_words = ['morning', 'afternoon', 'evening', 'night', 'today', 'yesterday', 'tomorrow']
        if any(word in sentence.lower() for word in time_words):
            scene_info['time_mentioned'] = True

        # Action vs dialogue detection
        if '"' in sentence or "'" in sentence:
            scene_info['action_type'] = 'dialogue'
        elif any(word in sentence.lower() for word in ['walked', 'ran', 'moved', 'went', 'came']):
            scene_info['action_type'] = 'action'
        else:
            scene_info['action_type'] = 'narrative'

        return scene_info

class EnhancedCoreferenceResolver:
    """Enhanced coreference resolver with neural models and advanced techniques"""

    def __init__(self, use_neural_coref: bool = True):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise

        # Initialize enhanced character memory
        self.character_memory = EnhancedCharacterMemory()

        # Neural coreference resolution
        self.use_neural_coref = use_neural_coref and HAS_NEURALCOREF
        if self.use_neural_coref:
            try:
                neuralcoref.add_to_pipe(self.nlp)
                logging.info("✓ Neural coreference resolution enabled")
            except Exception as e:
                logging.warning(f"Failed to initialize neuralcoref: {e}")
                self.use_neural_coref = False

        # AllenNLP coreference model (alternative)
        self.allen_coref = None
        if HAS_ALLENNLP and not self.use_neural_coref:
            try:
                self.allen_coref = Predictor.from_path(
                    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
                )
                logging.info("✓ AllenNLP coreference model loaded")
            except Exception as e:
                logging.warning(f"Failed to load AllenNLP coreference model: {e}")

        # Enhanced patterns and rules
        self.pronouns = {
            'personal': ['he', 'she', 'they', 'it'],
            'possessive': ['his', 'her', 'their', 'its'],
            'object': ['him', 'her', 'them', 'it'],
            'reflexive': ['himself', 'herself', 'themselves', 'itself']
        }

        # Character extraction patterns
        self.character_patterns = [
            r'\b(?:Mr|Mrs|Ms|Dr|Lord|Lady|Sir|Dame|King|Queen|Prince|Princess|Duke|Duchess)\.\s*(\w+(?:\s+\w+)*)',
            r'\b(Captain|General|Admiral|Colonel|Major|Lieutenant)\s+(\w+(?:\s+\w+)*)',
            r'\bthe\s+(Emperor|Empress|King|Queen)\s+(\w+(?:\s+\w+)*)',
            r'"([^"]*?)"\s+said\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+said[,:]?',
            r'(\w+(?:\s+\w+)*)\s+(?:whispered|shouted|replied|answered|asked)',
        ]

    def extract_characters_with_neural_ner(self, text: str) -> Dict[str, int]:
        """Enhanced character extraction using neural NER and patterns"""
        doc = self.nlp(text)
        characters = {}
        voice_id = 2  # Start from 2 (1 is narrator)

        # Extract person entities with neural NER
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip()) > 1:
                name = self._clean_entity_name(ent.text.strip())
                if name and self._is_valid_character_name(name):
                    if name not in characters:
                        characters[name] = voice_id

                        # Extract context around the entity
                        context = self._extract_entity_context(doc, ent)
                        self.character_memory.add_character(name, voice_id, context)
                        voice_id += 1

        # Extract characters using patterns
        pattern_characters = self._extract_characters_by_patterns(text)
        for name, context in pattern_characters.items():
            if name not in characters:
                characters[name] = voice_id
                self.character_memory.add_character(name, voice_id, context)
                voice_id += 1

        logging.info(f"Enhanced character extraction found {len(characters)} characters")
        return characters

    def _clean_entity_name(self, name: str) -> str:
        """Clean and normalize entity names"""
        # Remove common noise
        name = re.sub(r'^(The|A|An)\s+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name).strip()

        # Filter out common false positives
        false_positives = {
            'god', 'lord', 'sir', 'lady', 'king', 'queen', 'prince', 'princess',
            'chapter', 'part', 'book', 'volume', 'page', 'line', 'verse',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december'
        }

        if name.lower() in false_positives:
            return None

        return name

    def _is_valid_character_name(self, name: str) -> bool:
        """Validate if a name is likely a character"""
        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', name):
            return False

        # Must be reasonable length
        if len(name) < 2 or len(name) > 50:
            return False

        # Must not be all uppercase (likely abbreviation)
        if name.isupper() and len(name) > 3:
            return False

        # Must not contain numbers (usually not character names)
        if re.search(r'\d', name):
            return False

        return True

    def _extract_entity_context(self, doc, entity, window: int = 50) -> str:
        """Extract context around an entity"""
        start_idx = max(0, entity.start_char - window)
        end_idx = min(len(doc.text), entity.end_char + window)
        return doc.text[start_idx:end_idx]

    def _extract_characters_by_patterns(self, text: str) -> Dict[str, str]:
        """Extract characters using regex patterns"""
        characters = {}

        for pattern in self.character_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if groups:
                    # Take the last group as the character name
                    name = groups[-1].strip()
                    if self._is_valid_character_name(name):
                        # Get context around the match
                        start_idx = max(0, match.start() - 100)
                        end_idx = min(len(text), match.end() + 100)
                        context = text[start_idx:end_idx]
                        characters[name] = context

        return characters

    def resolve_sentence_coreferences_neural(self, sentence: str, sentence_idx: int) -> Tuple[str, List[str]]:
        """Resolve coreferences using neural models"""
        doc = self.nlp(sentence)
        resolved_sentence = sentence
        characters_in_sentence = []

        # Neural coreference resolution
        if self.use_neural_coref and doc._.has_coref:
            # Get coreference clusters
            for cluster in doc._.coref_clusters:
                main_mention = cluster.main
                for mention in cluster.mentions:
                    if mention != main_mention:
                        # Replace mention with main mention
                        resolved_sentence = resolved_sentence.replace(
                            mention.text, main_mention.text, 1
                        )

        # Extract characters from resolved sentence
        resolved_doc = self.nlp(resolved_sentence)
        for ent in resolved_doc.ents:
            if ent.label_ == "PERSON":
                name = self._clean_entity_name(ent.text.strip())
                if name and name in self.character_memory.characters:
                    if name not in characters_in_sentence:
                        characters_in_sentence.append(name)

        # Fallback to manual pronoun resolution
        if resolved_sentence == sentence:  # No neural resolution occurred
            resolved_sentence, additional_chars = self._manual_pronoun_resolution(
                sentence, sentence_idx
            )
            characters_in_sentence.extend(additional_chars)

        # Update character memory
        sentence_embedding = self.character_memory.sentence_embedder.encode(resolved_sentence)
        self.character_memory.update_context(
            resolved_sentence, sentence_idx, characters_in_sentence, sentence_embedding
        )

        return resolved_sentence, characters_in_sentence

    def _manual_pronoun_resolution(self, sentence: str, sentence_idx: int) -> Tuple[str, List[str]]:
        """Manual pronoun resolution fallback"""
        doc = self.nlp(sentence)
        resolved_sentence = sentence
        characters_found = []

        replacements = []

        for token in doc:
            if token.text.lower() in [p for pronoun_list in self.pronouns.values() for p in pronoun_list]:
                # Try to resolve pronoun
                sentence_embedding = self.character_memory.sentence_embedder.encode(sentence)
                resolved_char = self.character_memory.get_character_by_pronoun(
                    token.text, sentence_idx, sentence, sentence_embedding
                )

                if resolved_char:
                    replacements.append((token.idx, token.idx + len(token.text), resolved_char))
                    if resolved_char not in characters_found:
                        characters_found.append(resolved_char)

        # Apply replacements in reverse order
        for start_idx, end_idx, replacement in reversed(replacements):
            resolved_sentence = resolved_sentence[:start_idx] + replacement + resolved_sentence[end_idx:]

        return resolved_sentence, characters_found

    def resolve_text_coreferences(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Enhanced coreference resolution for entire text"""
        # First pass: extract characters with enhanced NER
        characters = self.extract_characters_with_neural_ner(text)
        logging.info(f"Enhanced extraction found {len(characters)} characters: {list(characters.keys())}")

        # Second pass: resolve coreferences sentence by sentence
        import nltk
        sentences = nltk.sent_tokenize(text)
        resolved_sentences = []

        for idx, sentence in enumerate(sentences):
            if self.use_neural_coref or self.allen_coref:
                resolved_sentence, chars_in_sentence = self.resolve_sentence_coreferences_neural(
                    sentence, idx
                )
            else:
                resolved_sentence, chars_in_sentence = self._manual_pronoun_resolution(
                    sentence, idx
                )
                # Update context for manual resolution
                self.character_memory.update_context(sentence, idx, chars_in_sentence)

            resolved_sentences.append(resolved_sentence)

            # Log significant coreference resolutions
            if resolved_sentence != sentence:
                logging.info(f"Enhanced resolution in sentence {idx}: {chars_in_sentence}")

        resolved_text = " ".join(resolved_sentences)

        # Update final character mapping
        final_characters = {}
        for char_name, char_info in self.character_memory.characters.items():
            final_characters[char_name] = char_info['voice_id']

        return resolved_text, final_characters

    def get_enhanced_character_info(self) -> Dict:
        """Return comprehensive character analysis"""
        return {
            name: {
                'voice_id': info['voice_id'],
                'aliases': info['aliases'],
                'mention_count': info['mention_count'],
                'last_mentioned': info['last_mentioned'],
                'importance_score': info['importance_score'],
                'emotional_state': info['emotional_state'],
                'relationships': info['relationships'],
                'role': info['role'],
                'inferred_gender': self.character_memory._infer_character_gender(name, info)
            }
            for name, info in self.character_memory.characters.items()
        }

def main():
    """Test enhanced coreference resolution"""
    logging.basicConfig(level=logging.INFO)

    resolver = EnhancedCoreferenceResolver()

    # Test text with complex coreferences
    test_text = """
    Sarah walked into the room where John was waiting. She looked tired from the long journey.
    "How are you feeling?" he asked her gently. The young woman smiled at him.
    Dr. Smith entered and greeted both of them. The doctor had important news to share.
    Sarah and John listened carefully to what the elderly physician had to say.
    """

    resolved_text, characters = resolver.resolve_text_coreferences(test_text)

    print("Original text:")
    print(test_text)
    print("\nResolved text:")
    print(resolved_text)
    print("\nCharacter mapping:")
    print(characters)
    print("\nEnhanced character analysis:")
    print(json.dumps(resolver.get_enhanced_character_info(), indent=2))

if __name__ == "__main__":
    main()