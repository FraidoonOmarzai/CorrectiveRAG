# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port your application runs on
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py"]
