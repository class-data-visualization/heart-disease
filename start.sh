#!/bin/bash
mkdir -p /root/.kaggle
cp kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json
streamlit run app.py --server.port 8000 --server.address 0.0.0.0
