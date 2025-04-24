# $ /usr/local/bin/python3 docchat.py

import os
import readline
from dotenv import load_dotenv
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import fitz 
import base64
from PIL import Image

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

load_dotenv()

def llm(messages, temperature=1):
    '''
    This function is my interface for calling the LLM.
    >>> llm([
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'user', 'content': 'What is the capital of France?'},
    ...     ], temperature=0)
    'The capital of France is Paris!'
    '''
    import groq
    client = groq.Groq()

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=temperature,
    )
    return chat_completion.choices[0].message.content


def chunk_text_by_words(text, max_words=100, overlap=50):
    """
    Splits text into overlapping chunks by word count.

    Parameters:
        text (str): The input text to split.
        max_words (int): Maximum number of words per chunk.
        overlap (int): Number of overlapping words between chunks.

    Returns:
        List[str]: A list of text chunks.

    Examples:
        >>> chunk_text_by_words("The quick brown fox jumps over the lazy dog", max_words=5, overlap=2)
        ['The quick brown fox jumps', 'fox jumps over the lazy', 'the lazy dog']

        >>> chunk_text_by_words("This is a short sentence", max_words=3, overlap=1)
        ['This is a', 'a short sentence', 'sentence']

        >>> chunk_text_by_words("One two", max_words=5, overlap=2)
        ['One two']

        >>> chunk_text_by_words("", max_words=5, overlap=2)
        []
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks

def score_chunk(chunk: str, query: str, language: str = "english") -> float:
    """
    Scores a chunk of text against a query using Jaccard similarity between 
    sets of lemmatized words with stopwords removed.

    Examples:
        >>> round(score_chunk("The sun is bright and hot.", "How hot is the sun?"), 2)
        0.67
        >>> round(score_chunk("The red car is speeding down the road.", "What color is the car?"), 2)
        0.2
        >>> score_chunk("Bananas are yellow.", "How do airplanes fly?")
        0.0
        >>> score_chunk("", "Is this empty?")
        0.0
        >>> score_chunk("Some random sentence", "")
        0.0
    """
    def preprocess(text):
        tokens = word_tokenize(text.lower(), preserve_line=True)
        stop_words = set(stopwords.words(language)) if language in stopwords.fileids() else set()
        lemmatizer = WordNetLemmatizer()
        processed = set(
            lemmatizer.lemmatize(word) for word in tokens
            if word.isalpha() and word not in stop_words
        )
        return processed

    chunk_words = preprocess(chunk)
    query_words = preprocess(query)

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words

    return len(intersection) / len(union)

def load_text(filepath_or_url):
    """
    Loads text from a local file or a URL. Supports plain text, HTML, and PDF formats.
    Examples:
        >>> text = load_text("docs/constitution-mx.txt")
        >>> "Constitución" in text
        True

        >>> text = load_text("docs/research_paper.pdf")
        >>> "Abstract" in text  # Replace with any real word in the PDF
        True

        >>> text = load_text("https://example.com")
        >>> "Example Domain" in text
        True
    """
    def is_url(path):
        try:
            result = urlparse(path)
            return result.scheme in ('http', 'https')
        except:
            return False

    def extract_pdf_text(file_path):
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text

    def extract_html_text(html):
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=' ', strip=True)

    if is_url(filepath_or_url):
        response = requests.get(filepath_or_url)
        response.raise_for_status()
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            return extract_html_text(response.text)
        elif 'application/pdf' in content_type:
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            return extract_pdf_text("temp.pdf")
        else:
            return response.text
    
    else:
        if not os.path.exists(filepath_or_url):
            raise FileNotFoundError(f"No such file: '{filepath_or_url}'")
        ext = os.path.splitext(filepath_or_url)[1].lower()
        if ext == '.pdf':
            return extract_pdf_text(filepath_or_url)  
        else:
            with open(filepath_or_url, 'rb') as f:
                return f.read().decode('utf-8', errors='ignore')

def find_relevant_chunks(text, query, num_chunks=5, max_words=50, overlap=10):
    """
    Splits the text into chunks and returns the most relevant chunks based on
    similarity to the query.

    Examples:
        >>> text = "Python is a programming language. It is used for data science. Java is also popular."
        >>> result = find_relevant_chunks(text, "What is Python?", num_chunks=1)
        >>> round(result[0][1], 2)
        0.14
        >>> "Python is a programming language" in result[0][0]
        True
    """
    chunks = chunk_text_by_words(text, max_words=max_words, overlap=overlap)
    scored_chunks = [(chunk, score_chunk(chunk, query)) for chunk in chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:num_chunks]

import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python docchat.py <file_path_or_url>")
        sys.exit(1)

    file_path_or_url = sys.argv[1]
    full_text = load_text(file_path_or_url)
    
    summary_prompt = [
    {"role": "system", "content": "Summarize this document in a few sentences."},
    {"role": "user", "content": full_text[:3000]}  # send only the first 3,000 characters
    ]

    summary = llm(summary_prompt)
    print(summary)

    messages = [{
        "role": "system",
        "content": "You are a helpful assistant that answers questions based on the provided document."
    }]

    while True:
        query = input("docchat> ")
        top_chunks = find_relevant_chunks(full_text, query, num_chunks=3)
        context = f"{summary}\n\n" + "\n".join([chunk for chunk, _ in top_chunks]) + f"\n\nQuestion: {query}"
        messages.append({"role": "user", "content": context})
        result = llm(messages[-2:])
        messages.append({"role": "assistant", "content": result})
        print(result)
