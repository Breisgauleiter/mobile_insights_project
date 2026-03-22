FROM node:20-slim

# Install Python 3 and OpenCV system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      libgl1 libglib2.0-0 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Node.js dependencies
COPY server/package.json server/package-lock.json* ./server/
RUN cd server && npm ci --omit=dev

# Install Python dependencies
COPY ml/requirements.txt ./ml/
RUN pip3 install --no-cache-dir --break-system-packages -r ml/requirements.txt

# Copy application code
COPY server/ ./server/
COPY ml/ ./ml/

RUN mkdir -p server/uploads

EXPOSE 3000

WORKDIR /app/server
CMD ["node", "index.js"]
