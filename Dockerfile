# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements first (for caching layers)
COPY requirements.txt /app/requirements.txt

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Install specific packages if not in requirements.txt
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Expose the port the app runs on
EXPOSE 5000

# Define the entrypoint and command to run the application
ENTRYPOINT ["python3"]
CMD ["app.py"]
