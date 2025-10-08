# 🔄 NLP File Converter

A **Streamlit web app** to convert translation and localization files between formats used in **CAT tools** (Trados, memoQ) and **NLP/LLM training pipelines** (Hugging Face, CSV, JSON).

Supports **English, French, and Chinese** with smart normalization, character conversion, and quality filtering.

![Demo](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

## ✨ Features

- **Convert between**:
  - **Translation Memory**: `.tmx`, `.sdltm`
  - **Bilingual Files**: `.xliff`, `.sdlxliff`, `.csv`, `.xlsx`
  - **Training Formats**: Hugging Face JSON, clean CSV/Excel
- **Language-aware processing** for **en**, **fr**, **zh**:
  - Normalize codes: `en-US` → `en`, `zh-CN` → `zhs`, `zh-TW` → `zht`
  - Optional **Simplified ↔ Traditional Chinese** conversion
- **Smart post-processing**:
  - Sentence splitting (spaCy + fallback)
  - Quality filtering (length ratio, script validation)
  - Deduplication
- **User control**:
  - Choose conversion purpose: *Translation* vs *Model Training*
  - Select output columns
  - Filter by language pair
  - Batch upload support

## 🚀 Try It Online

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-username-streamlit-app.streamlit.app)

> 🔜 *Deploy your own instance using Streamlit Community Cloud (free!)*

## 🛠️ Local Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/nlp-file-converter.git
   cd nlp-file-converter