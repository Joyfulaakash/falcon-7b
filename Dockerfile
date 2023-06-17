
FROM python:3.9

# Install dependencies
RUN pip install transformers einops accelerate langchain bitsandbytes

# Set CUDA visible devices
ENV CUDA_VISIBLE_DEVICES="0"

# Set maximum memory for GPUs
ENV MAX_MEMORY=""

# Copy the code into the container
COPY . /app
WORKDIR /app

# Run the API
CMD ["python", "api.py"]
EXPOSE 5000