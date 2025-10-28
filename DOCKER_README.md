# ISL Demo - Docker Setup

This directory contains Docker configuration files for the ISL (Indian Sign Language) Demo application.

## Files Created

- `Dockerfile` - Optimized multi-stage Docker configuration (Debian-based)
- `Dockerfile.alpine` - Ultra-optimized Alpine Linux version (smallest size)
- `docker-compose.yml` - Docker Compose configuration for easy deployment
- `.dockerignore` - Files to exclude from Docker build context
- `optimize-videos.sh` - Script to compress video files for smaller image size
- `SIZE_OPTIMIZATION.md` - Comprehensive guide on reducing Docker image size

## Prerequisites

1. **Docker** and **Docker Compose** installed on your system
2. **Google API Key** for Gemini (required for ISL gloss generation)

## Quick Start

### 1. Set up environment variables

Create a `.env` file in this directory:

```bash
# Google API Key for Gemini (required)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Set to "0" to use HF Whisper instead of Faster-Whisper
USE_FASTER_WHISPER=1
```

### 2. Build and run with Docker Compose

```bash
# Build and start the application (Debian-based, optimized)
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# For Alpine version (smallest size), modify docker-compose.yml:
# Uncomment the Alpine build section and comment the regular build line
```

### 3. Access the application

Open your browser and go to: http://localhost:8501

## Manual Docker Commands

If you prefer to use Docker directly:

```bash
# Build the optimized Debian image
docker build -t isl-demo .

# Build the Alpine image (smallest)
docker build -f Dockerfile.alpine -t isl-demo-alpine .

# Run the container (Debian)
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_google_api_key_here \
  -v $(pwd)/video_files:/app/video_files:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/uploads:/app/uploads \
  isl-demo

# Run the container (Alpine - smallest)
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_google_api_key_here \
  -v $(pwd)/video_files:/app/video_files:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/uploads:/app/uploads \
  isl-demo-alpine
```

## Features

- **Audio Transcription**: Uses Faster-Whisper or HF Whisper for speech-to-text
- **ISL Gloss Generation**: Converts English text to ISL gloss using Google Gemini
- **Video Concatenation**: Combines video clips to create ISL sequences
- **FFmpeg Integration**: Built-in FFmpeg for video processing
- **Persistent Storage**: Output and upload directories are mounted as volumes

## Image Size Optimization

The Docker setup includes multiple optimized configurations for different use cases:

### 1. Multi-stage Debian Build (`Dockerfile`) - WITH VIDEO FILES
- **Size**: ~2.5-3.5GB (includes video files)
- **Base**: Python 3.11-slim
- **Optimizations**:
  - Multi-stage build separates build and runtime dependencies
  - Virtual environment isolation
  - Removed unnecessary build tools from final image
  - Non-root user for security
  - Video files included in image for portability

### 2. Alpine Linux Build (`Dockerfile.alpine`) - WITH VIDEO FILES
- **Size**: ~1.8-2.5GB (smallest with videos included)
- **Base**: Python 3.11-alpine
- **Optimizations**:
  - Alpine Linux base (much smaller than Debian)
  - Multi-stage build with virtual environment
  - Minimal runtime dependencies
  - Non-root user for security
  - Video files included for maximum portability

### 3. Volume-mounted Videos (Dynamic Size)
- **Image Size**: ~0.5-1GB (without videos)
- **Videos**: Mounted from host
- **Best for**: Development and frequent updates

## Directory Structure

```
isl_demo/
├── Dockerfile              # Optimized multi-stage Docker configuration
├── Dockerfile.alpine       # Ultra-optimized Alpine version
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Docker ignore file
├── app.py                  # Main Streamlit application
├── audio_transcription.py  # Audio processing module
├── requirements.txt        # Python dependencies
├── video_files/            # ISL video clips (mounted as read-only)
├── output/                 # Generated videos (persistent)
└── uploads/                # Uploaded files (persistent)
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GOOGLE_API_KEY` | Google API key for Gemini | - | Yes |
| `USE_FASTER_WHISPER` | Use Faster-Whisper instead of HF Whisper | 1 | No |

## Reducing Image Size with Videos

### Option 1: Compress Videos Before Building (Recommended)

```bash
# Step 1: Optimize video files (reduces size by 50-70%)
./optimize-videos.sh

# Step 2: Review and apply
./optimize-videos.sh --apply

# Step 3: Build with compressed videos
docker build -f Dockerfile.alpine -t isl-demo-small .
```

### Option 2: Use Volume Mounting (Smallest Image)

Edit `docker-compose.yml` to uncomment the video_files volume:

```yaml
volumes:
  - ./video_files:/app/video_files:ro  # Uncomment this line
```

This creates a ~0.5-1GB image without videos.

### Option 3: Use Alpine Linux

```bash
docker build -f Dockerfile.alpine -t isl-demo-alpine .
```

See `SIZE_OPTIMIZATION.md` for detailed strategies.

## Troubleshooting

### Common Issues

1. **Google API Key not set**: Make sure to set the `GOOGLE_API_KEY` environment variable
2. **FFmpeg not found**: The Dockerfile includes FFmpeg, but if you encounter issues, check the logs
3. **Video files not found**: 
   - If using images with videos: Video files are included in the image
   - If using volumes: Ensure the `video_files` directory exists on host
4. **Port already in use**: Change the port mapping in `docker-compose.yml` if port 8501 is occupied
5. **Large image size**: Use `optimize-videos.sh` to compress videos before building

### Logs

```bash
# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f isl-demo
```

### Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove images and volumes
docker-compose down --rmi all --volumes
```

## Development

For development, you can mount the source code as a volume:

```yaml
# Add to docker-compose.yml under volumes
- ./app.py:/app/app.py
- ./audio_transcription.py:/app/audio_transcription.py
```

This allows you to make changes to the code without rebuilding the Docker image.
