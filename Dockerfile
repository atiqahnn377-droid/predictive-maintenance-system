# ===============================
# BASE IMAGE
# ===============================
FROM python:3.10-slim

# ===============================
# WORKING DIRECTORY
# ===============================
WORKDIR /app

# ===============================
# COPY REQUIREMENTS
# ===============================
COPY requirements.txt .

# ===============================
# INSTALL DEPENDENCIES
# ===============================
RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# COPY PROJECT FILES
# ===============================
COPY . .

# ===============================
# EXPOSE FLASK PORT
# ===============================
EXPOSE 5000

# ===============================
# RUN APPLICATION
# ===============================
CMD ["python", "api/app.py"]