# Search Engine System

A comprehensive information retrieval system built with Python, featuring advanced query processing, BM25 scoring, and intelligent snippet generation.

## Overview

This search engine system provides a complete solution for document indexing, searching, and result presentation. It supports multiple tokenization methods (Jieba and THULAC), advanced query syntax, and generates intelligent snippets for search results. The system is designed with performance optimization and scalability in mind.

## Project Structure

```
test/
├── app.py                    # Flask web application entry point
├── client.py                 # Client interface for testing
├── crawler.py                # Web crawler for data collection
├── filter.py                 # Content filtering utilities
├── personaltest.py           # Personal testing scripts
├── query.py                  # Core query engine with BM25 and SDM algorithms
├── query_optimized.py        # Performance-optimized query engine
├── search_engine.py          # Search engine wrapper with singleton pattern
├── search_processor.py       # Document processing and indexing (Jieba)
├── smart_snippet.py          # Intelligent snippet generation
├── thulac_pocessor.py        # Document processing with THULAC tokenization
├── url_normalization.py      # URL processing utilities
├── data/                     # Data storage directory
│   ├── jieba/               # Jieba-based processed data
│   └── thu/                 # THULAC-based processed data
├── final_txt/               # Processed text documents
├── static/                  # Static web assets
├── templates/               # HTML templates
└── tempHTML/                # Temporary HTML files
```

## Core Features

### 1. Advanced Query Processing
- **BM25 Scoring**: Industry-standard probabilistic ranking function
- **Sequential Dependence Model (SDM)**: Enhanced BM25 with term proximity features
- **Advanced Query Syntax**: Support for `inurl:`, `site:`, exact phrases, and OR operations
- **Dual Tokenization**: Support for both Jieba and THULAC Chinese word segmentation

### 2. Document Processing
- **HTML to Text Conversion**: Clean extraction of textual content
- **Inverted Index Construction**: Efficient term-document mapping with position information
- **Batch Processing**: Scalable processing of large document collections
- **Multiple Format Support**: HTML and plain text document processing

### 3. Intelligent Features
- **Smart Snippet Generation**: Context-aware excerpt generation with query term highlighting
- **Query Optimization**: Caching and performance enhancements
- **Flexible Configuration**: Customizable parameters for different use cases

### 4. Web Interface
- **Flask-based Web App**: User-friendly search interface
- **Real-time Search**: Fast query processing and result display
- **Responsive Design**: Modern web interface with CSS styling

## Design Philosophy

### 1. Modular Architecture
The system follows a modular design pattern where each component has a specific responsibility:
- **Query Engine**: Handles search logic and ranking algorithms
- **Processors**: Manage document indexing and preprocessing
- **Snippet Generator**: Provides intelligent result summarization
- **Web Layer**: Offers user interface and API endpoints

### 2. Performance Optimization
- **Singleton Pattern**: Global query engine instance to avoid repeated initialization
- **Caching Mechanisms**: LRU cache for frequently accessed data
- **Batch Processing**: Efficient handling of large document sets
- **Optimized Data Structures**: Fast lookup and retrieval operations

### 3. Extensibility
- **Plugin Architecture**: Easy integration of new tokenizers and ranking algorithms
- **Configuration-driven**: Flexible parameter tuning without code changes
- **Multiple Backend Support**: Support for different data storage formats

### 4. Algorithm Implementation
- **BM25**: Classic probabilistic ranking with tunable parameters (k1, b)
- **SDM Enhancement**: Incorporates term proximity through ordered and unordered windows
- **Position-aware Indexing**: Maintains term position information for advanced features

## Usage Instructions

### 1. Environment Setup

```bash
# Install required dependencies
pip install flask jieba thulac dill numpy beautifulsoup4

# Ensure data directories exist
mkdir -p data/jieba data/thu final_txt tempHTML
```

### 2. Data Preparation

```python
# Process documents with Jieba tokenization
from search_processor import SearchEngineProcessor
processor = SearchEngineProcessor()
processor.process_all()

# Or process with THULAC tokenization
from thulac_pocessor import SearchEngineProcessorTHU
processor_thu = SearchEngineProcessorTHU()
processor_thu.process_all()
```

### 3. Running the Web Application

```bash
# Start the Flask web server
python app.py

# Access the search interface at http://localhost:5000
```

### 4. Programmatic Usage

```python
# Initialize query engine
from query import QueryEngine
engine = QueryEngine(thu=False)  # Use Jieba tokenization
# engine = QueryEngine(thu=True)   # Use THULAC tokenization

# Perform search
results = engine.retrieval_url("your search query", k=10)

# Advanced search with custom parameters
scores = engine.enhanced_bm25_scores(
    engine.inverted_index, 
    engine.doc_length, 
    "query", 
    engine.N,
    k=10,
    lambda_t=0.7,  # Unigram weight
    lambda_o=0.2,  # Ordered window weight
    lambda_u=0.1   # Unordered window weight
)
```

### 5. Advanced Query Syntax

```
# URL filtering
inurl:python programming

# Site restriction
site:github.com machine learning

# Multiple sites
site:github.com OR site:stackoverflow.com

# Exact phrase matching
"machine learning algorithms"

# Directory-specific search
site:https://docs.python.org/3/ functions
```

### 6. Configuration Options

```python
# BM25 parameters
k1 = 1.5      # Term frequency saturation parameter
b = 0.75      # Length normalization parameter

# SDM weights
lambda_t = 0.7  # Unigram feature weight
lambda_o = 0.2  # Ordered window feature weight
lambda_u = 0.1  # Unordered window feature weight

# Window size for proximity features
window_size = 8

# Snippet generation
window_chars = 200  # Character length for snippets
```

## Performance Considerations

- **Memory Usage**: The system loads inverted indices into memory for fast access
- **Initialization Time**: First query may take longer due to data loading
- **Scalability**: Designed to handle thousands of documents efficiently
- **Caching**: Implements multiple levels of caching for optimal performance

## Dependencies

- **Flask**: Web framework for the user interface
- **Jieba**: Chinese word segmentation
- **THULAC**: Alternative Chinese tokenizer
- **Dill**: Enhanced pickle for data serialization
- **NumPy**: Numerical computations
- **BeautifulSoup4**: HTML parsing and processing

## Contributing

The system is designed for extensibility. Key areas for enhancement:
- Additional ranking algorithms
- Support for more languages
- Advanced query features
- Performance optimizations
- User interface improvements

---

*This search engine system demonstrates modern information retrieval techniques with practical implementation considerations for real-world deployment.*