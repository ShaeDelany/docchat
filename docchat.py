import os
import readline
from dotenv import load_dotenv

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


def chunk_text_by_words(text, max_words=5, overlap=2):
    """
    Splits text into overlapping chunks by word count.
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
    Scores a chunk against a query using Jaccard similarity of lemmatized sets.
    Works best for English.
    """

    def preprocess(text):
        tokens = word_tokenize(text.lower())
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


if __name__ == '__main__':
    messages = []
    messages.append({
        "role": "system",
        "content": "Your are a helpful assistant. You always speak like a pirate. You always answer in one sentence"
    })
    while True:
        text = input('docchat> ')
        messages.append({
            'role': 'user',
            'content': text,
        })
        result = llm(messages)

        messages.append({
            'role': 'assistant',
            'content': result
        })
        print('result=', result)
        import pprint
        pprint.pprint(messages)
