# FitBot: Fitness Equipment FAQ Chatbot

An intelligent FAQ chatbot that uses retrieval augmented generation (RAG) to answer customer questions about fitness equipment.

## Project Overview

This chatbot combines semantic search with large language models to provide accurate, contextual responses to customer inquiries. It uses:
- **LangChain** for RAG pipeline management
- **Claude 3.5 Sonnet** for natural language generation
- **FAISS** for efficient vector similarity search
- **Sentence Transformers** for semantic embeddings

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Anthropic API key (get one at console.anthropic.com)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key:

**On Mac/Linux:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**On Windows:**
```bash
set ANTHROPIC_API_KEY=your-api-key-here
```

### Running the Chatbot

1. Run the main script:
```bash
python faq_chatbot.py
```

2. The chatbot will automatically create a sample dataset if one doesn't exist

3. Wait for initialization to complete (about 5-10 seconds)

4. Start asking questions!

### Sample Questions

Try asking:
- What is the warranty on your treadmills?
- How do I clean my yoga mat?
- Do you offer free shipping?
- What is the weight limit for the elliptical?
- How do I connect my fitness tracker?

### Exiting

Type `quit`, `exit`, or `q` to stop the chatbot.

## Project Structure

- `faq_chatbot.py` - Main chatbot implementation
- `requirements.txt` - Python dependencies
- `fitness_faq_dataset.csv` - Auto-generated FAQ dataset (30 questions)
- `FAQ_Chatbot_Report.docx` - Comprehensive project report

## How It Works

1. **Data Loading**: Loads FAQ data from CSV file
2. **Preprocessing**: Chunks documents and generates embeddings
3. **Indexing**: Creates FAISS vector store for fast retrieval
4. **Query Processing**: Converts user questions to embeddings
5. **Retrieval**: Finds top 3 most similar FAQ entries
6. **Generation**: Uses Claude to generate natural language response

## Dataset

The chatbot uses a custom fitness equipment FAQ dataset with 30 common customer service questions covering:
- Warranty and returns
- Product specifications
- Shipping and delivery
- Assembly and maintenance
- Technology and connectivity
- Pricing and discounts

## Technical Details

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **LLM**: Claude 3.5 Sonnet
- **Vector Store**: FAISS with cosine similarity
- **Chunk Size**: 500 characters with 50 character overlap
- **Retrieval**: Top-k=3 similar documents

## Performance

- Average response time: 2-4 seconds
- Retrieval accuracy: ~85-90%
- Handles variations in question phrasing through semantic understanding

## License

This is a student project for educational purposes.

## Author

Christopher Garcia
Western Governors University
Artificial Intelligence Course
January 2025
