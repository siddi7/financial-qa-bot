# financial-qa-bot
AI-powered financial statement analyzer using Google Gemini
## Features

- PDF financial statement processing
- Natural language question answering
- Vector-based document search
- Real-time financial data extraction
- Batch processing for large documents
- Performance monitoring and optimization
- Caching system for improved response times

## Prerequisites

- Docker and Docker Compose
- Python 3.10 or higher (for local development)
- Google Gemini API key
- Pinecone API key

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/siddi7/financial-qa-bot.git
cd financial-qa-bot
```

2. Create a .env file with your API keys:
```bash
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

4. Access the application at http://localhost:7860
