#!/bin/bash

# Script to optimize video files for smaller Docker image size
# This script compresses video files to reduce the overall image size

echo "🎬 Video Optimization Script for ISL Demo"
echo "=========================================="

VIDEO_DIR="video_files"
OPTIMIZED_DIR="video_files_optimized"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ Error: ffmpeg is not installed. Please install ffmpeg first."
    echo "   On macOS: brew install ffmpeg"
    echo "   On Ubuntu: sudo apt-get install ffmpeg"
    exit 1
fi

# Create optimized directory if it doesn't exist
mkdir -p "$OPTIMIZED_DIR"

# Get total size before optimization
TOTAL_SIZE_BEFORE=$(du -sh "$VIDEO_DIR" | cut -f1)
echo "📊 Original size: $TOTAL_SIZE_BEFORE"

echo "🔄 Compressing video files..."
FILES_COUNTED=0
FILES_PROCESSED=0

# Process all MP4 files in the video_files directory
for file in "$VIDEO_DIR"/*.mp4; do
    if [ -f "$file" ]; then
        ((FILES_COUNTED++))
        filename=$(basename "$file")
        
        # Compress video with optimized settings for smaller size
        # Using libx264 with veryslow preset for maximum compression
        # Reducing bitrate and resolution slightly to save space
        echo "  Processing: $filename"
        
        ffmpeg -i "$file" \
            -c:v libx264 \
            -preset veryslow \
            -crf 28 \
            -c:a copy \
            -movflags +faststart \
            -y \
            "$OPTIMIZED_DIR/$filename" > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            ((FILES_PROCESSED++))
        fi
    fi
done

# Get total size after optimization
TOTAL_SIZE_AFTER=$(du -sh "$OPTIMIZED_DIR" | cut -f1)
echo "📊 Optimized size: $TOTAL_SIZE_AFTER"
echo "✅ Processed $FILES_PROCESSED/$FILES_COUNTED files"

echo ""
echo "🎯 Optimization Options:"
echo "1. Backup original and replace with optimized: ./optimize-videos.sh --apply"
echo "2. Compare sizes to decide: ls -lh video_files/ video_files_optimized/"
echo "3. Use original files as-is: docker build -f Dockerfile"
echo ""
echo "To apply optimizations, run: ./optimize-videos.sh --apply"

# Apply optimizations if --apply flag is passed
if [ "$1" == "--apply" ]; then
    echo ""
    echo "⚠️  Applying optimizations..."
    backup_dir="video_files_backup_$(date +%Y%m%d_%H%M%S)"
    echo "📦 Creating backup in: $backup_dir"
    cp -r "$VIDEO_DIR" "$backup_dir"
    
    echo "🔄 Replacing with optimized files..."
    rm -rf "$VIDEO_DIR"
    mv "$OPTIMIZED_DIR" "$VIDEO_DIR"
    
    echo "✅ Done! Original files backed up in: $backup_dir"
fi
