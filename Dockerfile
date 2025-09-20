# Start from a slim Python base
FROM python:3.11-slim
RUN apt-get update && apt-get install -y sudo
RUN adduser --disabled-password --gecos '' myuser
RUN usermod -aG sudo myuser
RUN echo 'myuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER myuser
# Install uv
RUN sudo pip install --no-cache-dir uv

# Set working directory
WORKDIR /code

# Copy project files
COPY . .

# Install dependencies using uv
RUN sudo uv pip install --system -r requirements.txt

# Expose FastAPI port for Hugging Face
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
