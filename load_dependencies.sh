#!/bin/bash

echo "🚀 Setting up the Grooming Detection Project..."

# Update package list
echo "🔄 Updating package list..."
sudo apt update -y

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt install -y ffmpeg

# Create and activate virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install required Python libraries
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete! Run 'source venv/bin/activate' to activate the virtual environment."
