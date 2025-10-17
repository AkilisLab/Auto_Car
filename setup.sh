#!/bin/bash

# Create a virtual environment
python3 -m venv --system-site-packages car_env

# Activate the virtual environment
source car_env/bin/activate

# Install packages
pip install smbus opencv-python pyaudio 

# Deactivate the virtual environment
# deactivate

echo "All packages installed in car_env"