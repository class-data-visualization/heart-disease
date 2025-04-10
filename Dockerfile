# Use official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app files
COPY . .

# Make the startup script executable
RUN chmod +x start.sh

# Expose the port Streamlit will run on
EXPOSE 8000

# Run the start script
CMD ["./start.sh"]