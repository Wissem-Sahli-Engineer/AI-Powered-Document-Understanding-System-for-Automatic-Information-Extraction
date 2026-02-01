#!/bin/bash

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend
echo "Starting Streamlit frontend..."
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Wait for all processes to finish
wait
