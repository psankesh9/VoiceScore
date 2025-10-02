# VoiceScore: Advanced Multi-Character Text-to-Speech

VoiceScore is an advanced text-to-speech system that converts web novels into audiobooks with distinct character voices using Named Entity Recognition (NER), coreference resolution, and multi-speaker TTS synthesis.

## Features

🎭 **Multi-Character Voice Synthesis**
- Automatically detects character names using NER
- Assigns unique voices to each character
- Resolves pronouns to character names using coreference resolution

🧠 **Advanced Character Memory**
- Tracks character context across the entire text
- Handles character aliases and alternative names
- Maintains character relationships and emotional states

🎯 **Smart Voice Assignment**
- Creates voice profiles based on character descriptions
- Uses FAISS vector database for voice similarity matching
- Supports custom voice descriptions for characters

🚀 **Modern TTS Technology**
- Integrates GlowTTS/VITS for high-quality speech synthesis
- Supports both CPU and GPU acceleration
- Batch processing for efficient audio generation

## System Architecture

```
Web Novel URL
    ↓
[Web Scraping] → Raw Text
    ↓
[Text Preprocessing] → Clean Text
    ↓
[NER + Coreference Resolution] → Character Mapping
    ↓
[Voice Profile Creation] → Character Voice Profiles
    ↓
[Voice Assignment] → Sentence-Character Assignments
    ↓
[TTS Synthesis] → Individual Audio Segments
    ↓
[Audio Organization] → Complete Audiobook
```

## Installation

1. **Run the setup script**
   ```bash
   python setup.py
   ```

2. **Manual installation (if needed)**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Usage

### Run the Complete Pipeline

```bash
python voicescore_pipeline.py
```

### Custom Voice Descriptions

Edit the `custom_voices` dictionary in `voicescore_pipeline.py`:

```python
custom_voices = {
    "Character Name": "voice description (e.g., young, cheerful female voice)",
    "Another Character": "confident, deep male voice with slight accent"
}
```

## Output Structure

```
output_directory/
├── raw_scraped_content.txt     # Original scraped text
├── resolved_text.txt           # Text with resolved coreferences
├── voice_profiles.json         # Character voice profiles
├── processing_report.txt       # Detailed processing report
├── audio_playlist.m3u         # Audio playlist file
└── audio_segments/            # Individual audio files
    ├── segment_0001_Narrator.wav
    ├── segment_0002_Character.wav
    └── ...
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA support (optional, for GPU acceleration)
- Internet connection (for downloading models)

## Key Components

1. **advanced_coreference.py**: Character tracking and pronoun resolution
2. **advanced_tts_system.py**: Multi-speaker TTS synthesis
3. **voicescore_pipeline.py**: Main processing pipeline
4. **setup.py**: Dependency installation script

## Next Steps

1. Run `python setup.py` to install dependencies
2. Edit the URL in `voicescore_pipeline.py` if needed
3. Run `python voicescore_pipeline.py` to process your web novel
4. Check the output directory for generated audio files

The system will automatically:
- Scrape the web page
- Extract character names
- Resolve pronouns to characters
- Assign unique voices
- Generate audio for each sentence
- Create a complete audiobook with character voices