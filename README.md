<div align="center">

# 🏥 Medical Research Assistant

### AI-Powered Research Tool Using RAG (Retrieval-Augmented Generation)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/Powered%20by-Groq%20AI-orange.svg)](https://groq.com/)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Keys](#-api-keys)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [How It Works](#-how-it-works)
- [Roadmap](#-roadmap)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**Medical Research Assistant** is an intelligent document analysis system that helps researchers, medical professionals, and students quickly find answers from large collections of research papers. Using state-of-the-art **Retrieval-Augmented Generation (RAG)** technology, it combines:

- 📄 **Document Processing** - Extract and analyze text from multiple PDFs
- 🔍 **Smart Retrieval** - TF-IDF-based semantic search across thousands of sentences
- 🤖 **AI Generation** - LLM-powered answer synthesis with source citations
- 🌐 **Live Research** - Direct integration with PubMed database

### Why This Project?

Traditional research requires:
- ❌ Hours of manual reading
- ❌ Searching through multiple papers
- ❌ Difficulty in synthesizing information
- ❌ No way to verify sources quickly

**Our solution provides:**
- ✅ Instant answers from your research papers
- ✅ Automatic source citation
- ✅ Evidence-based synthesis from multiple sources
- ✅ User-friendly web interface

---

## ✨ Features

### 🎯 Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-PDF Processing** | Upload and analyze multiple research papers simultaneously |
| **TF-IDF Search** | Advanced relevance ranking using Term Frequency-Inverse Document Frequency |
| **RAG Architecture** | Combines retrieval and generation for accurate, cited answers |
| **PubMed Integration** | Search and analyze papers directly from PubMed database |
| **Source Tracking** | Every answer includes citations with page numbers |
| **Web UI** | Beautiful, intuitive Streamlit interface |

### 🚀 Advanced Features

- **Three Input Methods**
  - 📁 Upload local PDF files
  - 🔗 Download PDFs from URLs
  - 🔬 Search PubMed automatically

- **Smart Analysis**
  - Sentence-level extraction
  - Relevance scoring
  - Top-N result ranking
  - Context-aware generation

- **Professional Output**
  - Synthesized answers (not just excerpts)
  - Source citations in academic format
  - Relevance scores for transparency
  - Downloadable results

---

## 📸 Demo

### Command-Line Interface
```bash
🏥 Medical Research Assistant v2.2

💬 Ask your research question: What is heat surge in India?

✅ Found 15 PDF files
📄 Processing papers...
✅ Total sentences: 8,012

🔍 Analyzing with TF-IDF...

📌 ANSWER:
Based on the research, heat surges in India refer to extreme temperature 
events that pose significant health risks including heat exhaustion, 
heatstroke, and mental health impacts [Sources 1, 2, 3]...

📚 SOURCES USED:
1. heat_health_india_2023.pdf (Page 11) - Score: 0.7749
2. mental_health_heat.pdf (Page 23) - Score: 0.6628
```

### Web Interface

**Main Dashboard:**
![Main Interface](screenshots/main_dashboard.png)

**Search Results:**
![Search Results](screenshots/results_1.png)

![Search Results](screenshots/results_2.png)

**AI-Generated Answer:**
![AI Answer](screenshots/results_3.png)

---

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                            │
│  (Streamlit Web App / Command Line Interface)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  INPUT PROCESSING                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Local PDFs   │  │  URL Download│  │ PubMed API   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 DOCUMENT PROCESSING                          │
│  • PDF Text Extraction (PyPDF2)                             │
│  • Sentence Segmentation                                     │
│  • Metadata Tracking (source, page)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  RETRIEVAL LAYER                             │
│  • TF-IDF Vectorization (scikit-learn)                      │
│  • Cosine Similarity Calculation                            │
│  • Top-N Ranking                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 GENERATION LAYER                             │
│  • Context Building                                          │
│  • Prompt Engineering                                        │
│  • LLM API Call (Groq - Llama 3.3 70B)                      │
│  • Answer Synthesis                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUT                                   │
│  • Synthesized Answer with Citations                        │
│  • Source References                                         │
│  • Relevance Scores                                          │
└─────────────────────────────────────────────────────────────┘
```

### RAG Pipeline
```python
# Simplified workflow
question = "What is heat surge?"
    ↓
pdfs = load_pdfs(folder)              # 1. Load documents
    ↓
sentences = extract_sentences(pdfs)   # 2. Segment into sentences
    ↓
relevant = tfidf_search(question)     # 3. Retrieve top matches
    ↓
answer = llm_generate(relevant)       # 4. Generate answer
    ↓
display(answer + citations)           # 5. Show with sources
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Groq API key ([Get it free](https://console.groq.com))

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/medical-research-assistant.git
cd medical-research-assistant
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## 🚀 Usage

### Option 1: Web Interface (Recommended)
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Steps:**
1. Enter your Groq API key in the sidebar
2. Choose data source (Upload PDFs / Search PubMed)
3. Provide your research papers
4. Enter your question
5. Click "Search & Analyze"
6. Get AI-powered answers with citations!

### Option 2: Command-Line Interface
```bash
python main.py
```

**Interactive prompts:**
```
🔑 Enter your Groq API key: gsk_...
📚 Choose input method:
1. Local folder
2. Download from URLs
3. Search PubMed
Enter choice: 1

📂 Enter folder path: ./research_papers
💬 Ask your question: What is heat surge in India?
```

---

## 🔑 API Keys

### Getting a Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create new API key
5. Copy the key (starts with `gsk_...`)

**Note:** Keep your API key secure. Never commit it to version control.

### Environment Variables (Optional)

Create a `.env` file:
```bash
GROQ_API_KEY=your_api_key_here
PUBMED_EMAIL=your_email@example.com
```

Load in code:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
```

---

## 📁 Project Structure
```
medical-research-assistant/
│
├── app.py                      # Streamlit web interface
├── main.py                     # Command-line interface
├── requirements.txt            # Python dependencies
├── README.md                   # This file                     
│
├── screenshots/                # Demo images
│   ├── main_interface.png
│   ├── search_results.png
│   └── ai_answer.png
│
├── examples/                   # Example usage
│   ├── sample_questions.txt
│   └── sample_pdfs/
│       └── example_paper.pdf
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   └── CONTRIBUTING.md
│
└── tests/                      # Unit tests
    ├── test_pdf_processing.py
    ├── test_tfidf_search.py
    └── test_rag_pipeline.py
```

---

## 🔧 Technologies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **Streamlit** | 1.28+ | Web interface framework |
| **PyPDF2** | 3.0+ | PDF text extraction |
| **scikit-learn** | 1.3+ | TF-IDF vectorization |
| **Groq** | 0.4+ | LLM API client |
| **Biopython** | 1.81+ | PubMed API integration |
| **NLTK** | 3.8+ | Natural language processing |

### Why These Technologies?

- **Streamlit**: Fastest way to build data apps in Python
- **TF-IDF**: Industry-standard retrieval algorithm, lightweight and fast
- **Groq**: Ultra-fast LLM inference (500+ tokens/sec)
- **PyPDF2**: Reliable PDF parsing without external dependencies
- **Biopython**: Official PubMed API wrapper

---

## ⚙️ How It Works

### 1. Document Processing
```python
def load_all_pdfs_with_tracking(folder_path):
    """
    Loads all PDFs from a folder and tracks:
    - Source filename
    - Page number
    - Extracted text
    """
    for pdf_file in pdf_files:
        reader = PdfReader(filepath)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            sentences = text.split('.')
            # Store with metadata
```

**Output:**
```python
[
    {'text': 'Heat waves cause health impacts...', 
     'source': 'paper1.pdf', 
     'page': 5},
    {'text': 'India experiences extreme heat...', 
     'source': 'paper2.pdf', 
     'page': 12},
    ...
]
```

### 2. TF-IDF Retrieval
```python
def tfidf_search(question, document_sentences, top_n=5):
    """
    Uses TF-IDF to find most relevant sentences
    
    TF-IDF measures:
    - Term Frequency: How often a word appears in a sentence
    - Inverse Document Frequency: How rare/important a word is
    
    Formula: TF-IDF = TF × IDF
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',  # Remove common words
        ngram_range=(1, 2)     # Consider 1 and 2-word phrases
    )
    
    # Convert text to numbers
    tfidf_matrix = vectorizer.fit_transform([question] + texts)
    
    # Calculate similarity
    similarities = cosine_similarity(question_vector, sentence_vectors)
    
    # Return top matches
    return sorted_results[:top_n]
```

**Why TF-IDF?**
- ✅ Fast (no GPU needed)
- ✅ Interpretable scores
- ✅ Works offline
- ✅ Industry-proven for 30+ years

### 3. RAG Generation
```python
def generate_research_answer(question, retrieved_sentences, api_key):
    """
    Builds a prompt with:
    1. User question
    2. Retrieved context from documents
    3. Instructions for the LLM
    """
    
    # Build context from top sentences
    context = "\n".join([
        f"[Source {i}]: {sentence['text']}"
        for i, sentence in enumerate(retrieved_sentences)
    ])
    
    # Create prompt
    prompt = f"""
    Question: {question}
    Context: {context}
    
    Instructions:
    - Synthesize information from all sources
    - Cite sources using [Source 1], [Source 2]
    - Provide evidence-based answer
    """
    
    # Call LLM
    answer = groq_client.generate(prompt)
    return answer
```

**LLM Model:** Llama 3.3 70B (via Groq)
- 70 billion parameters
- 500+ tokens/second inference speed
- Trained on medical and scientific literature

---

## 🗺️ Roadmap

### ✅ Completed (v1.0 - v2.2)

- [x] PDF text extraction
- [x] TF-IDF search implementation
- [x] RAG pipeline
- [x] Groq API integration
- [x] PubMed integration
- [x] Web UI (Streamlit)
- [x] Source citation tracking
- [x] Multi-PDF support

### 🚧 In Progress (v2.3)

- [ ] Embeddings-based search (better than TF-IDF)
- [ ] Conversation history
- [ ] Export results to PDF/DOCX
- [ ] User authentication


### Development Setup
```bash
# Fork and clone the repo
git clone https://github.com/yourusername/medical-research-assistant.git

# Create a branch
git checkout -b feature/your-feature-name

# Make changes and test
python -m pytest tests/

# Commit and push
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

```


## 🙏 Acknowledgments

### Inspiration & Resources

- **Anthropic** - For Claude and RAG architecture inspiration
- **Groq** - For blazing-fast LLM inference
- **Streamlit** - For making Python web apps accessible
- **PubMed/NCBI** - For providing free access to medical literature
- **scikit-learn** - For robust ML tools

