
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


def fetch_content_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text


class PDFQA:
    def __init__(self, pdf_path=None):
        if pdf_path:
            self.text = self._convert_pdf_to_text(pdf_path)
            self.train(self.text)

    def _convert_pdf_to_text(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text

    def train_on_url(self, url):
        web_text = fetch_content_from_url(url)
        self.train(web_text)

    def train(self, text):
        self.text = text
        self.sentences = re.split(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.sentences)

    def get_answer(self, query):
        query_vec = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(
            query_vec, self.vectors).flatten()
        matched_index = np.argmax(cosine_similarities)
        if cosine_similarities[matched_index] > 0.05:
            return self.sentences[matched_index]
        else:
            return "No relevant information found."


pdf_qa = PDFQA()
url = "https://home.iitd.ac.in/about.php"
pdf_qa.train_on_url(url)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/ask', methods=['POST'])
def ask():
    query = request.form['query']
    answer = pdf_qa.get_answer(query)
    return {'answer': answer}


if __name__ == "__main__":
    app.run(debug=True)
