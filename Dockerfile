FROM python:3.12-slim

WORKDIR  /app

COPY . .

RUN pip install -r requirements.txt

RUN python -c "import nltk; \
nltk.download('stopwords'); \
nltk.download('punkt'); \
nltk.download('punkt_tab')"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8502"]