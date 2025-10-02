"""
Advanced Coreference Resolution System for VoiceScore
Resolves pronouns to character names and tracks character context across text
"""

import os
import json
import logging
import spacy
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import re

# +
# #!pip install sentence_transformers
# -

class CharacterMemory:
    """Manages character information and context across the text"""

    def __init__(self, max_context_window=10):
        self.characters = {}  # character_name -> character_info
        self.context_window = deque(maxlen=max_context_window)
        self.sentence_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.character_mentions = defaultdict(list)  # tracks where each character appears

    def add_character(self, name: str, voice_id: int, context: str = ""):
        """Add or update character information"""
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
                'relationships': {}
            }

        self.characters[name]['mention_count'] += 1
        return self.characters[name]

    def add_alias(self, character_name: str, alias: str):
        """Add alternative names/aliases for a character"""
        if character_name in self.characters:
            if alias not in self.characters[character_name]['aliases']:
                self.characters[character_name]['aliases'].append(alias)

    def get_character_by_pronoun(self, pronoun: str, sentence_idx: int, context: str) -> Optional[str]:
        """Resolve pronoun to character name based on context and recency"""
        pronoun_lower = pronoun.lower()

        # Gender-based pronoun mapping
        male_pronouns = {'he', 'him', 'his', 'himself'}
        female_pronouns = {'she', 'her', 'hers', 'herself'}
        neutral_pronouns = {'they', 'them', 'their', 'themselves'}

        candidates = []

        # Find recently mentioned characters
        for char_name, char_info in self.characters.items():
            if self.character_mentions[char_name]:
                last_mention = max(self.character_mentions[char_name])

                # Prioritize recently mentioned characters
                if sentence_idx - last_mention <= 5:  # within 5 sentences
                    recency_score = 1.0 / (sentence_idx - last_mention + 1)

                    # Simple gender heuristics (can be improved with NER features)
                    gender_score = 0.5  # default
                    if any(word in char_info['context'].lower() for word in ['mr.', 'sir', 'king', 'prince', 'duke']):
                        gender_score = 1.0 if pronoun_lower in male_pronouns else 0.1
                    elif any(word in char_info['context'].lower() for word in ['ms.', 'mrs.', 'lady', 'queen', 'princess', 'duchess']):
                        gender_score = 1.0 if pronoun_lower in female_pronouns else 0.1

                    candidates.append((char_name, recency_score * gender_score))

        # Return the most likely candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def update_context(self, sentence: str, sentence_idx: int, characters_in_sentence: List[str]):
        """Update context window and character mentions"""
        self.context_window.append({
            'sentence': sentence,
            'index': sentence_idx,
            'characters': characters_in_sentence
        })

        # Update character mention tracking
        for char_name in characters_in_sentence:
            self.character_mentions[char_name].append(sentence_idx)
            if char_name in self.characters:
                self.characters[char_name]['last_mentioned'] = sentence_idx

class AdvancedCoreferenceResolver:
    """Advanced coreference resolution using spaCy + custom logic"""

    def __init__(self):
        # Load spaCy model with NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise

        self.character_memory = CharacterMemory()

        # Pronoun patterns
        self.pronouns = {
            'personal': ['he', 'she', 'they', 'it'],
            'possessive': ['his', 'her', 'their', 'its'],
            'object': ['him', 'her', 'them', 'it'],
            'reflexive': ['himself', 'herself', 'themselves', 'itself']
        }

        # Common character titles and descriptors
        self.character_titles = [
            r'\b(?:Mr|Mrs|Ms|Dr|Lord|Lady|Sir|Dame|King|Queen|Prince|Princess|Duke|Duchess)\.\s*(\w+)',
            r'\b(Captain|General|Admiral|Colonel|Major|Lieutenant)\s+(\w+)',
            r'\bthe\s+(Emperor|Empress|King|Queen)\s+(\w+)',
        ]

    def extract_characters_from_text(self, text: str) -> Dict[str, int]:
        """Extract character names using NER and assign voice IDs"""
        doc = self.nlp(text)
        characters = {}
        voice_id = 2  # Start from 2 (1 is narrator)

        # Extract person entities
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip()) > 1:
                name = ent.text.strip()
                if name not in characters:
                    characters[name] = voice_id

                    # Add character to memory with context
                    context_start = max(0, ent.start_char - 100)
                    context_end = min(len(text), ent.end_char + 100)
                    context = text[context_start:context_end]

                    self.character_memory.add_character(name, voice_id, context)
                    voice_id += 1

        # Extract characters using title patterns
        for pattern in self.character_titles:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 1:
                    name = match.group(-1)  # Last group is usually the name
                    if name and name not in characters:
                        characters[name] = voice_id

                        # Add full match as context
                        full_title = match.group(0)
                        context_start = max(0, match.start() - 100)
                        context_end = min(len(text), match.end() + 100)
                        context = text[context_start:context_end]

                        self.character_memory.add_character(name, voice_id, context)
                        self.character_memory.add_alias(name, full_title)
                        voice_id += 1

        return characters

    def resolve_sentence_coreferences(self, sentence: str, sentence_idx: int) -> Tuple[str, List[str]]:
        """Resolve coreferences in a single sentence"""
        doc = self.nlp(sentence)
        resolved_sentence = sentence
        characters_in_sentence = []

        # Find characters mentioned directly
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if name in self.character_memory.characters:
                    characters_in_sentence.append(name)
                else:
                    # Check aliases
                    for char_name, char_info in self.character_memory.characters.items():
                        if name in char_info['aliases']:
                            characters_in_sentence.append(char_name)
                            break

        # Replace pronouns with character names
        tokens_to_replace = []

        for token in doc:
            if token.text.lower() in [p for pronoun_list in self.pronouns.values() for p in pronoun_list]:
                # Try to resolve pronoun
                resolved_char = self.character_memory.get_character_by_pronoun(
                    token.text, sentence_idx, sentence
                )

                if resolved_char:
                    tokens_to_replace.append((token.idx, token.idx + len(token.text), resolved_char))
                    if resolved_char not in characters_in_sentence:
                        characters_in_sentence.append(resolved_char)

        # Apply replacements (in reverse order to maintain indices)
        for start_idx, end_idx, replacement in reversed(tokens_to_replace):
            resolved_sentence = resolved_sentence[:start_idx] + replacement + resolved_sentence[end_idx:]

        # Update character memory
        self.character_memory.update_context(sentence, sentence_idx, characters_in_sentence)

        return resolved_sentence, characters_in_sentence

    def resolve_text_coreferences(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Resolve coreferences in entire text and return character mapping"""

        # First pass: extract all characters
        characters = self.extract_characters_from_text(text)
        logging.info(f"Extracted {len(characters)} characters: {list(characters.keys())}")

        # Second pass: resolve coreferences sentence by sentence
        import nltk
        sentences = nltk.sent_tokenize(text)
        resolved_sentences = []

        for idx, sentence in enumerate(sentences):
            resolved_sentence, chars_in_sentence = self.resolve_sentence_coreferences(sentence, idx)
            resolved_sentences.append(resolved_sentence)

            # Log if coreferences were resolved
            if resolved_sentence != sentence:
                logging.info(f"Resolved coreferences in sentence {idx}: {chars_in_sentence}")

        resolved_text = " ".join(resolved_sentences)

        # Update character mapping with any new discoveries
        final_characters = {}
        for char_name, char_info in self.character_memory.characters.items():
            final_characters[char_name] = char_info['voice_id']

        return resolved_text, final_characters

    def get_character_info(self) -> Dict:
        """Return detailed character information"""
        return {
            name: {
                'voice_id': info['voice_id'],
                'aliases': info['aliases'],
                'mention_count': info['mention_count'],
                'last_mentioned': info['last_mentioned']
            }
            for name, info in self.character_memory.characters.items()
        }

def main():
    """Test the coreference resolution system"""
    resolver = AdvancedCoreferenceResolver()

    # Test text
    test_text = """
    Sarah walked into the room. She was carrying a heavy bag.
    John greeted her warmly. He had been waiting for Sarah for hours.
    They sat down together and began their conversation.
    """

    logging.basicConfig(level=logging.INFO)

    resolved_text, characters = resolver.resolve_text_coreferences(test_text)

    print("Original text:")
    print(test_text)
    print("\nResolved text:")
    print(resolved_text)
    print("\nCharacter mapping:")
    print(characters)
    print("\nCharacter details:")
    print(json.dumps(resolver.get_character_info(), indent=2))

if __name__ == "__main__":
    main()


