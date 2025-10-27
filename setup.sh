#!/bin/bash
set -e

# Install Python packages
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup completed successfully!"
