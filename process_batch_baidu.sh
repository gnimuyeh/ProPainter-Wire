#!/bin/bash

# Persistent BaiduPCS-Go binary path
BAIDU_PCS="/workspace/BaiduPCS-Go-v3.9.7-linux-amd64/BaiduPCS-Go"

# Remote Netdisk folder (input + output in same place)
REMOTE_FOLDER=""

# Local working directories
LOCAL_INPUT_DIR="/workspace/workdata/input_videos"
LOCAL_OUTPUT_DIR="/workspace/workdata/propainter_results"

mkdir -p "$LOCAL_INPUT_DIR"
mkdir -p "$LOCAL_OUTPUT_DIR"

# Get file list from remote folder
echo "📁 Listing files in $REMOTE_FOLDER..."
ALL_FILES=$($BAIDU_PCS ls "$REMOTE_FOLDER" | awk '/\.mov/ {print $5}')


for FILE in $ALL_FILES; do
    # Skip if file is a mask or result file
    if [[ "$FILE" == *_mask.* || "$FILE" == *_result.* ]]; then
        continue
    fi

    # Parse base and extension
    FILENAME=$(basename "$FILE")
    BASENAME="${FILENAME%.*}"
    EXT="${FILENAME##*.}"

    MASK_NAME="${BASENAME}_mask.${EXT}"
    RESULT_NAME="${BASENAME}_result.${EXT}"

    INPUT_PATH="$LOCAL_INPUT_DIR/$FILENAME"
    MASK_PATH="$LOCAL_INPUT_DIR/$MASK_NAME"
    OUTPUT_PATH="$LOCAL_OUTPUT_DIR/$RESULT_NAME"

    if [[ -f "$OUTPUT_PATH" ]]; then
        echo "✅ Skipping $FILENAME — local result file already exists."
        # Optionally, upload it if you want, or just skip fully
        continue
    fi
    
    # Check if result file already exists in cloud
    if $BAIDU_PCS search -path "$REMOTE_FOLDER" "$RESULT_NAME" | grep -q "文件总数: 1"; then
        echo "✅ Skipping $FILENAME — result exists."
        continue
    fi

    # Check if mask file exists in cloud
    if ! $BAIDU_PCS search -path "$REMOTE_FOLDER" "$MASK_NAME" | grep -q "文件总数: 1"; then
        echo "❌ Skipping $FILENAME — no mask file found."
        continue
    fi

    # Download input and mask
    echo "⬇️ Downloading $FILENAME and $MASK_NAME..."
    $BAIDU_PCS download "$REMOTE_FOLDER/$FILENAME" --saveto "$LOCAL_INPUT_DIR"
    $BAIDU_PCS download "$REMOTE_FOLDER/$MASK_NAME" --saveto "$LOCAL_INPUT_DIR"

    # Run ProPainter inference
    echo "🧠 Running ProPainter on $FILENAME..."
    python inference_propainter.py \
        --subvideo_length 10 \
        --raft_iter 50 \
        --ref_stride 10 \
        --mask_dilation 0 \
        --neighbor_length 10 \
        --video "$INPUT_PATH" \
        --mask "$MASK_PATH" \
        --output "$OUTPUT_PATH" \
        --save_masked_in

    # Upload result
    # echo "📤 Uploading $RESULT_NAME..."
    # $BAIDU_PCS upload "$OUTPUT_PATH" "$REMOTE_FOLDER/"

    # Cleanup
    # rm "$INPUT_PATH" "$MASK_PATH" "$OUTPUT_PATH"
done

echo "✅ All done!"
