# DocuMind AI Backend

FastAPI backend for advanced document processing with Google Gemini AI.

## Features

- Advanced RAG system
- Google Gemini integration
- HackRX webhook compatibility
- Multiple document format support

## Deployment

### Railway Deployment

1. Connect your GitHub repository to Railway
2. Add environment variables:
   - `GOOGLE_API_KEY`: Your Google AI API key
   - `MODEL_NAME`: `gemini-1.5-flash` (optional)
3. Railway will automatically detect the configuration from `railway.toml`

## Environment Variables

- GOOGLE_API_KEY
- MODEL_NAME=gemini-1.5-flash
