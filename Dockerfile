# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local application files to the container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN conda install pytorch torchvision cpuonly -c pytorch

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python3", "app.py"]

