#!/bin/bash
# Setup Kaggle credentials
mkdir -p ~/.kaggle/
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Run the Streamlit app on port 8000, accessible externally
streamlit run app.py --server.port=8000 --server.address=0.0.0.0
