# ResearchAI v2.0: Interactive Document Knowledge Extraction System

A production-ready Streamlit application for analyzing research papers and documents using AI-powered retrieval-augmented generation (RAG).

## Features

### Core Functionality
- **Multi-Format Document Support**: PDF, DOCX, TXT, HTML, Markdown
- **Visual Dashboard**: Instant insights with charts, statistics, and key findings
- **AI-Powered Q&A**: Ask questions about your documents with source citations
- **Streaming Responses**: Real-time response generation for better UX
- **Smart Chunking**: Section-aware text splitting for better retrieval

### Dashboard Features
- Document statistics (word count, reading time, sentences, pages)
- Section distribution visualization
- Readability score with Flesch Reading Ease
- AI-generated summary
- Key findings extraction
- Top keywords visualization
- Suggested research questions
- Named entity detection

### Advanced Features
- **Multi-LLM Support**: OpenAI GPT-4/3.5, Anthropic Claude
- **Source Citations**: See exactly which parts of the document answered your question
- **Dark/Light Theme**: Customizable UI theme
- **Vector Store Persistence**: Save and reload document embeddings
- **Analytics Tracking**: Usage statistics and query tracking
- **Export Options**: Export chat history to Markdown

## Project Structure

```
NLP_Project/
├── app_v2.py                 # Main application (v2.0)
├── app.py                    # Legacy application (v1.0)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py       # Application settings
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Multi-format document processing
│   │   ├── llm_service.py         # LLM and vector store operations
│   │   └── analytics.py           # Usage analytics
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Data models and schemas
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── text_processing.py    # Text cleaning and analysis
│   │   └── visualization.py      # Chart data generation
│   ├── components/
│   │   ├── __init__.py
│   │   ├── dashboard.py      # Dashboard UI component
│   │   ├── chat.py           # Chat interface component
│   │   └── sidebar.py        # Sidebar component
│   └── templates/
│       ├── __init__.py
│       └── styles.py         # CSS and HTML templates
├── data/
│   ├── vector_stores/        # Persisted vector stores
│   └── analytics.db          # SQLite analytics database
└── tests/
    └── test_chunk.py         # Unit tests
```

## Installation

### Prerequisites
- Python 3.9 or newer
- pip package manager

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/NLP_Project.git
   cd NLP_Project
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv

   # Activate:
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key  # Optional
   ```

## Running the Application

### Version 2.0 (Recommended)
```bash
streamlit run app_v2.py
```

### Legacy Version 1.0
```bash
streamlit run app.py
```

The application will start at `http://localhost:8501`

## Usage Guide

### 1. Upload Documents
- Click the file uploader in the sidebar
- Select one or more documents (PDF, DOCX, TXT, HTML, MD)
- Click "Process" to analyze

### 2. Explore the Dashboard
After processing, the Dashboard tab shows:
- **Metrics**: Word count, reading time, sentences, pages
- **Summary**: AI-generated document summary
- **Key Findings**: Main contributions and findings
- **Section Distribution**: Visual breakdown of document sections
- **Readability Score**: How easy the document is to read
- **Keywords**: Most frequent terms
- **Suggested Questions**: AI-generated questions to explore

### 3. Ask Questions
Switch to the Chat tab to:
- Ask questions about document content
- Get AI-powered answers with source citations
- See which sections of the document were used
- Export chat history to Markdown

### 4. Customize Settings
In the sidebar settings:
- Switch between Light/Dark theme
- Select AI provider (OpenAI/Anthropic)
- Choose specific model
- Adjust response creativity
- Toggle source citations
- Enable/disable streaming

## Configuration

### Settings (src/config/settings.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | 1500 | Maximum characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `max_file_size_mb` | 50 | Maximum file size |
| `default_model` | gpt-3.5-turbo | Default LLM model |
| `enable_analytics` | True | Track usage analytics |

## API Keys

### OpenAI (Required)
1. Create account at [OpenAI](https://platform.openai.com)
2. Generate API key
3. Add to `.env` file

### Anthropic (Optional)
1. Create account at [Anthropic](https://console.anthropic.com)
2. Generate API key
3. Add to `.env` file

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

**Package Installation Errors**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**FAISS Installation Issues (Windows)**
```bash
pip install faiss-cpu
```

**GPU Support**
```bash
pip install faiss-gpu  # Instead of faiss-cpu
```

**API Key Errors**
- Ensure `.env` file exists in project root
- Check API key is valid and has credits

## Changelog

### v2.0.0
- Visual dashboard with document analytics
- Multi-format document support (PDF, DOCX, TXT, HTML, MD)
- Source citations in AI responses
- Streaming responses
- Dark/Light theme toggle
- Multi-LLM support (OpenAI, Anthropic)
- Analytics and usage tracking
- Modular architecture
- Export functionality

### v1.0.0
- Initial release
- PDF support
- Basic Q&A functionality

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Support

For issues and feature requests, please use the GitHub Issues page.
