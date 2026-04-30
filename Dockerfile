FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# ===============================
# EXPOSE FLASK PORT
# ===============================
EXPOSE 5000

# ===============================
# RUN APPLICATION
# ===============================
CMD ["python", "api/app.py"]