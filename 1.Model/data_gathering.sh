#!/bin/bash

# Define directories
DATA_DIR="data"
ARCHIVE_NAME="MedNIST.tar.gz"
URL="https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz"

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

# Download the tar.gz file
echo "Downloading MedNIST data..."
curl -L $URL -o $DATA_DIR/$ARCHIVE_NAME

# Extract the tar.gz file silently
echo "Extracting MedNIST data..."
tar -xzf $DATA_DIR/$ARCHIVE_NAME -C $DATA_DIR

# Clean up the tar.gz file
rm $DATA_DIR/$ARCHIVE_NAME

echo "Data gathering complete. Files are in the '$DATA_DIR' directory."
