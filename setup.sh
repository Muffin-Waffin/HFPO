#!/bin/bash

echo "Creating virtual environment..."

python3 -m venv .venv

source .venv/bin/activate

echo "Installing dependencies..."

pip install --upgrade pip

pip install -r requirements.txt

pip install -e .

echo "Setup complete!"

echo "Activate with:"
echo "source .venv/bin/activate"