# Use official Python 3.12 image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy your code into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port if you are using Flask (change if needed)
EXPOSE 10000

# Command to run your app (change if your main file has a different name)
CMD ["python", "app.py"]
