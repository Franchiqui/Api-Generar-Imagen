# Base image for Python environment (adjust as needed)
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-2024-05-06


RUN python -m venv venv

RUN /bin/bash -c "source venv/bin/activate"

# Copia el archivo requirements.txt y luego instala las dependencias de Python
COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt
# Copy your application code (replace with your actual path)
COPY . /app

# Working directory for your application
WORKDIR /app

# Crear el directorio de imágenes
RUN mkdir -p /app/images

# Expose the port for FastAPI application
EXPOSE 5080

# Command to execute your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5080"]
