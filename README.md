# 📖 Novel-TTS: AI-Powered Web Novel Audiobook Generation

## 📌 Overview

Novel-TTS is an AI-driven project that transforms web novels into high-quality audiobooks. By leveraging **Natural Language Processing (NLP)**, **Named Entity Recognition (NER)**, and **Text-to-Speech (TTS)** models, the system assigns unique voices to characters in novels while using a separate narrator voice for non-dialogue text. The ultimate goal is to make web novels—especially those without existing audiobook adaptations—more **accessible and immersive**.

## 🎯 Motivation

- **Bridging the Audiobook Gap for Translated Novels** – Many web novels, especially those translated from languages like Chinese, Japanese, and Korean, do not have official audiobooks. This project aims to fill that gap.
- **Enhancing Accessibility** – Readers with **visual impairments** often struggle to access web novels. By converting them into spoken content, we ensure that they can enjoy these stories as well.
- **Customization & Research** – This project lays the groundwork for **advanced voice customization**, allowing users to modify voices based on character descriptions and personalities. It also builds upon previous research in **machine translation and literary style adaptation**.
- **Future Audiobook System** – The long-term vision is a fully automated, **customizable audiobook platform** for web novels, integrating custom **TTS models trained for more natural and expressive speech**.

## ⚙ Features

### ✅ 1. Web Scraping
- Extracts full text from novel sites such as **ReadNovelFull** and saves it as a `.txt` file for processing.
- Uses **BeautifulSoup** and **Requests** to retrieve content while handling text formatting.

### ✅ 2. Named Entity Recognition (NER)
- Utilizes a **Hugging Face transformer model** (e.g., `dbmdz/bert-large-cased-finetuned-conll03-english`) to detect named entities in the text.
- Assigns each detected **name a unique voice ID**.
- Implements **regex-based matching** to track character mentions in sentences.

### ✅ 3. Voice Assignment for Characters & Narration
- The system determines whether a sentence contains a named entity:
  - **If a named entity (character) is present, it assigns the corresponding character's voice.**
  - **Otherwise, the sentence is read in the narrator's voice.**
- Ensures **seamless transitions between voices**.

### ✅ 4. Text-to-Speech (TTS) Integration
- Uses **pyttsx3** for initial TTS generation with different voices.
- Future iterations will incorporate more advanced TTS models (e.g., **Tacotron, VITS, or custom neural TTS solutions**).
- Plans for **voice customization** based on character attributes (**age, gender, tone**, etc.).

### ✅ 5. Flow of Execution

[Start] │ ▼ [Scrape webpage and extract text] │ ▼ [Segment text into sentences] │ ▼ [Run Named Entity Recognition (NER)] │ ▼ [Assign a unique voice ID to each named entity] │ ▼ [Determine if a sentence contains a named entity] │ ├──► Yes: Assign the corresponding character voice │ └──► No: Use the narrator's voice │ ▼ [Speak the sentence using the assigned voice] │ ▼ [Repeat for all sentences] │ ▼ [End]

## 🚀 Future Enhancements

- **Advanced TTS Models** – Implementing **custom neural TTS** trained on expressive speech to improve voice quality.
- **Character-Specific Voices** – Generating voices **based on textual character descriptions**.
- **Gender & Emotion Recognition** – Using **NLP to infer gender/emotion** for better voice selection.
- **Customizable Audiobooks** – Allowing users to **assign their own voices** or select preferred TTS models.
- **Multi-Language Support** – Expanding to work with **multiple languages**, leveraging research in **machine translation**.

## 📝 Installation Guide

### 1️⃣ Install Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install requests beautifulsoup4 transformers nltk pyttsx3
python novel_tts.py "https://readnovelfull.com/absolute-resonance/chapter-1481-four-sword-realm-goddess-divineflame-mirror.html"
```
This will:

Scrape the web novel content.
Run NER to identify characters.
Assign voices and generate speech output.
📚 Research & References
Machine Translation & Literary NLP – Previous work in bilingual text alignment and multilingual literary datasets.
Expressive TTS Models – Exploration of Tacotron, FastSpeech, and VITS for natural voice synthesis.
Audiobook Accessibility – Studies on making literature accessible to visually impaired users.
👥 Contributors
Developed as part of an ongoing research project into NLP, TTS, and digital storytelling.

📜 License
MIT License (or another open-source license of your choice).

📢 Contact & Updates
For future updates and discussions, reach out via GitHub or relevant forums.

🔹 Future Steps:

Test additional AI models such as Claude, Mistral, and DeepSeek.
Benchmark against industry standards using ComplexFuncBench.
Develop a fully customizable audiobook system with realistic character voices.
