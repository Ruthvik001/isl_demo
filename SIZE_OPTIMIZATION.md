# Docker Image Size Optimization Guide

This guide explains how to minimize Docker image size when including video files in your ISL demo application.

## 📊 Size Comparison

| Strategy | Image Size | Trade-offs |
|----------|------------|------------|
| With video files (no optimization) | ~4-5GB | Easiest to deploy |
| Optimized multi-stage build | ~2-3GB | Better caching |
| Alpine + optimized videos | ~1.5-2GB | Smaller size |
| Volume-mounted videos | ~0.5-1GB | Requires external storage |

## 🎯 Recommended Solutions

### Solution 1: Multi-stage Build with Video Files (Current Setup)

**Best for**: Production deployments with video files included

```bash
# Build the image
docker build -t isl-demo .

# Or with Alpine (smaller)
docker build -f Dockerfile.alpine -t isl-demo-alpine .
```

**Size**: ~2-3GB (Debian) or ~1.5-2GB (Alpine)

**Pros**:
- ✅ Videos included in image
- ✅ Portable (no external dependencies)
- ✅ Optimized layers for better caching

**Cons**:
- ❌ Larger image size
- ❌ Longer push/pull times

### Solution 2: Compress Video Files Before Building

**Best for**: Maximum size reduction with videos included

```bash
# Step 1: Optimize video files
./optimize-videos.sh

# Step 2: Review and apply optimizations
./optimize-videos.sh --apply

# Step 3: Build with optimized videos
docker build -t isl-demo-optimized .
```

**Estimated size reduction**: 50-70% of video files size

**Pros**:
- ✅ Videos included in image
- ✅ Smaller overall size
- ✅ Better compression

**Cons**:
- ❌ Slight quality reduction (usually imperceptible)
- ❌ Takes time to compress

### Solution 3: Volume Mount Videos (Smallest Image)

**Best for**: Local development and frequent updates

Edit `docker-compose.yml`:
```yaml
services:
  isl-demo:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      # Mount video files from host
      - ./video_files:/app/video_files:ro
      - ./output:/app/output
      - ./uploads:/app/uploads
```

**Size**: ~0.5-1GB (without video files in image)

**Pros**:
- ✅ Smallest image size
- ✅ Easy to update videos without rebuilding
- ✅ Shared storage across containers

**Cons**:
- ❌ Videos must be available on host
- ❌ Not portable (requires host files)
- ❌ Depends on external storage

### Solution 4: Separate Video Registry/Storage

**Best for**: Large-scale deployments

Use a separate storage solution:
- AWS S3 / Google Cloud Storage
- Docker Registry with separate repository
- Network file storage (NFS, CIFS)

Then mount or download at runtime.

## 🔧 Implementation Details

### Video Compression Settings

The `optimize-videos.sh` script uses these FFmpeg settings:

```bash
ffmpeg -i input.mp4 \
    -c:v libx264 \
    -preset veryslow \
    -crf 28 \
    -c:a copy \
    -movflags +faststart \
    output.mp4
```

**Settings explained**:
- `-c:v libx264`: H.264 codec (widely supported)
- `-preset veryslow`: Maximum compression (slower but smaller)
- `-crf 28`: Higher compression, acceptable quality
- `-c:a copy`: Keep audio codec (no re-encoding)
- `-movflags +faststart`: Optimize for streaming

### Build Arguments for Flexibility

You can add build arguments to Dockerfile:

```dockerfile
ARG INCLUDE_VIDEOS=true
COPY app.py .
COPY audio_transcription.py .

# Conditionally copy videos
COPY --chown=appuser:appuser video_files* ./video_files/
RUN if [ "$INCLUDE_VIDEOS" = "true" ]; then \
    echo "Videos included"; \
else \
    echo "Videos excluded - use volumes"; \
fi
```

Build with:
```bash
# With videos
docker build --build-arg INCLUDE_VIDEOS=true -t isl-demo .

# Without videos (use volumes)
docker build --build-arg INCLUDE_VIDEOS=false -t isl-demo .
```

## 📈 Build Performance Tips

### 1. Layer Caching Optimization

The current Dockerfile order ensures maximum cache hits:

```dockerfile
# These rarely change
COPY requirements.txt .
RUN pip install ...

# Copy videos last (often change)
COPY video_files/ ./video_files/
```

### 2. Parallel Builds

Use Docker BuildKit for parallel layer building:

```bash
DOCKER_BUILDKIT=1 docker build -t isl-demo .
```

### 3. Multi-platform Builds

If deploying to multiple platforms:

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 \
    -t isl-demo:latest .
```

## 🎬 Quick Reference

### Current Setup (Recommended)

```bash
# Standard build with videos
docker build -t isl-demo .

# Alpine build (smaller)
docker build -f Dockerfile.alpine -t isl-demo-alpine .

# With docker-compose
docker-compose up --build
```

### For Maximum Size Reduction

```bash
# 1. Optimize videos first
./optimize-videos.sh --apply

# 2. Build with optimized videos
docker build -f Dockerfile.alpine -t isl-demo-small .

# 3. Check size
docker images isl-demo-small
```

### For Development

```bash
# Use volumes (no videos in image)
docker-compose -f docker-compose.yml up

# Volume mounting handles videos externally
```

## 📦 Expected Sizes

Based on typical ISL video collections:

| Content | Size | Image Size (Debian) | Image Size (Alpine) |
|---------|------|-------------------|---------------------|
| Video files (~50MB) | 50MB | 2.5GB | 1.8GB |
| Video files (~200MB) | 200MB | 2.7GB | 2.0GB |
| Video files (~500MB) | 500MB | 3.0GB | 2.3GB |
| No videos (volumes) | 0MB | 2.0GB | 1.5GB |

## ✅ Best Practices Summary

1. **Use multi-stage builds** - Already implemented
2. **Optimize video files** - Use `optimize-videos.sh`
3. **Use Alpine for smaller base** - `Dockerfile.alpine`
4. **Layer efficiently** - Copy frequently changing files last
5. **Consider volumes** - For development and large videos
6. **Enable BuildKit** - For faster builds: `DOCKER_BUILDKIT=1`
7. **Remove unnecessary dependencies** - Minimal runtime deps
8. **Use .dockerignore** - Exclude unnecessary files

## 🚀 Recommended Production Setup

For production with video files included:

```bash
# 1. Optimize videos
./optimize-videos.sh --apply

# 2. Build optimized Alpine image
DOCKER_BUILDKIT=1 docker build -f Dockerfile.alpine -t isl-demo:latest .

# 3. Tag and push
docker tag isl-demo:latest your-registry/isl-demo:latest
docker push your-registry/isl-demo:latest

# 4. Deploy
docker-compose up -d
```

Expected final size: **~1.5-2GB** with videos included
