# Start from a slim Python base
FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /code

# Copy project files
COPY . .

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt

# Expose FastAPI port for Hugging Face
EXPOSE 7860

# Run FastAPI app with uvicorn
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
