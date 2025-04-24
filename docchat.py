import readline
from groq import Groq
import groq
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def llm(messages, temperature=1):
    '''
    This function is my interface for calling the LLM.
    The messages argument should be a list of dictionaries.

    >>> llm([
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'user', 'content': 'What is the capital of France?'},
    ...     ], temperature=0)
    'The capital of France is Paris!'
    '''
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

    Examples:
        >>> text = "The quick brown fox jumps over the lazy dog. It was a sunny day and the birds were singing."
        >>> chunks = chunk_text_by_words(text, max_words=5, overlap=2)
        >>> len(chunks)
        7
        >>> chunks[0]
        'The quick brown fox jumps'
        >>> chunks[1]
        'fox jumps over the lazy'
        >>> chunks[4]
        'sunny day and the birds'
        >>> chunks[-1]
        'singing.'
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


def load_spacy_model(language: str):
    """
    Loads a spaCy model for the specified language.
    """
    LANGUAGE_MODELS = {
        'french': 'fr_core_news_sm',
        'german': 'de_core_news_sm',
        'spanish': 'es_core_news_sm',
        'english': 'en_core_web_sm',
    }

    if language not in LANGUAGE_MODELS:
        raise ValueError(f"Unsupported language: {language}")

    return spacy.load(LANGUAGE_MODELS[language])


def score_chunk(chunk: str, query: str, language: str = "french") -> float:
    if language != "english":
        raise ValueError("Only English is supported with NLTK in this version.")

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        tokens = word_tokenize(text.lower())
        return set(
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha and token not in stop_words
        )

    chunk_words = preprocess(chunk)
    query_words = preprocess(query)

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words

    return len(intersection) / len(union)

if __name__=='__main__':
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
