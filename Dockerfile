# -----------------------------
# Base Image with Miniconda
# -----------------------------
FROM continuumio/miniconda3:latest

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    zbar-tools \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    git \
    && apt-get clean

# -----------------------------
# Create Working Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy Conda Environment File
# -----------------------------
COPY environment.yml .

# -----------------------------
# Create Conda Environment
# -----------------------------
# RUN conda create vi_env -f environment.yml
# conda env create -f environment.yml
RUN conda env create -f environment.yml

# Activate environment by default
ENV PATH /opt/conda/envs/ai-vision/bin:$PATH
ENV CONDA_DEFAULT_ENV=ai-vision

# -----------------------------
# Copy Application Code
# -----------------------------
COPY . .

# -----------------------------
# Expose FastAPI Port
# -----------------------------
EXPOSE 8000

# -----------------------------
# Command to Run FastAPI App
# -----------------------------
CMD ["fastapi", "dev", "verifyidentity.py", "--host", "0.0.0.0", "--port", "8000"]