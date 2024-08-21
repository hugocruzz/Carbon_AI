FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies step
RUN pip install --upgrade pip  # Ensure pip is up-to-date
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/main.py"]
