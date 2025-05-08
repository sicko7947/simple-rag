# Simple RAG Implementation

**Note: This is a simple starter implementation, not a complete production-ready design. It provides the basic architecture and components to help you get started with building your own RAG system.**

This repository contains a simple Retrieval-Augmented Generation (RAG) implementation designed to enhance language model responses with retrieved context.

## Overview

This RAG system includes components for:
- Data ingestion and document processing
- Vector storage and embedding with [Supabase](https://supabase.com)
- Semantic retrieval with guardrails
- Context building and prompt management
- Cognitive agent with knowledge state tracking
- User interfaces (CLI and web)

## System Architecture

### [1] Data Ingestion Layer
- **Multi-source Document Ingestion**: PDF / HTML / TEXT
- **Metadata Extraction**: Timestamps, Authors, Categories, Confidence Scores
- **Document Structuring & Embedding**: Optional HyDE Query Synthesis

### [2] Knowledge Vector Store
- **Vector Database**: Supabase pg_vector
- **Document Version Management**: Historical Changes, Retraction Markers
- **Weighting Mechanism**: Confidence Levels, Citation Counts, Internal Priority

### [3] Retrieval + Validation Engine (Partially Optimized)
- **Semantic Retrieval**: Top-K + Reranking (e.g., Cohere Reranker) 
- **Multi-document Conflict Detection**: Based on Similarity & Logical Contradictions
- **Query Expansion**: HyDE / Query Rewriting
- **Guardrail Validator**: Fact Cross-verification / Risk Control Rules

### [4] LLM Inference Layer
- **Context Builder**: Document Merging, Window Constraints
- **Prompt Templates**: Roles, Scenarios, Output Control
- **Response Generation**: LLMs like Claude, GPT, Mistral
- **Output Inspector**: Keywords + Structure + Compliance Assessment

### [5] Cognitive Agent Layer
- **Cognitive State Modeling**: "What I Know / Don't Know"
- **Conflict Detector**: Comparison with Knowledge Graph/Compliance Graph
- **Reasoning Chain Tracking**: Chain-of-Thought, Tool Use
- **Decision Trigger**: Whether to Suggest User Actions

### [6] User Experience + Logging
- **Multi-modal Input Support**: Voice / Image / Text
- **Conversation Traceability**: Question Sources, Response References
- **User Feedback System**: Error Detection / Adoption Tracking
- **Data Annotation & Fine-tuning Support**: RLHF / RLAIF

## Project Structure

- `src/agent/`: Cognitive agent implementation with knowledge state management
- `src/data_ingestion/`: Document loading, processing, and metadata extraction
- `src/language_model/`: Model interface, context building, and prompt management
- `src/retriever/`: Semantic retrieval with query expansion and guardrails
- `src/user_interface/`: CLI and web interface implementations
- `src/vector_store/`: Vector embedding and storage options


## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd simple-rag
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables by creating a `.env` file in the project root:
```bash
# Create .env file
cp .env.example .env

# Edit the .env file with your configuration
# Required variables include:
# - API keys for language models (OPENAI_API_KEY, etc.)
# - Supabase connection details (SUPABASE_URL, SUPABASE_KEY)
# - Other service configurations as needed
```

### Running the Web Interface

To start the web interface, run:
```bash
python -m src.user_interface.web
```

### Running the CLI Interface

Alternatively, you can use the command-line interface:
```bash
python -m src.user_interface.cli
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.