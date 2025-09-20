# API Assistant

This is a FastAPI backend that powers the AI Portfolio Assistant. It exposes REST APIs for chat, job matching, and user interaction logging.

## Features

- FastAPI backend with CORS enabled for frontend integration
- Integrates with Gemini/OpenAI LLMs for chat completions
- Reads profile summary and resume PDF for context
- Tool-call support for logging unknown questions and user details
- Endpoints:
  - `/api/hello` – Health check
  - `/api/ask` – Chat with the AI assistant
  - `api/match` - Match JD with Resume skills

## Setup

1. Install dependencies:
   ```sh
   uv pip install --system -r requirements.txt
   ```
2. Set up your `.env` file with `GOOGLE_API_KEY` and other secrets.
3. Place your resume PDF(s) in `me/` and summary in `me/summary.txt`.
4. Run locally:
   ```sh
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```
5. Docker support: use port 8000 fro local. 7860 is needed for huggingface
   ```sh
   docker build -t api-assistant .
   docker run -p 7860:7860 --env-file .env api-assistant
   ```