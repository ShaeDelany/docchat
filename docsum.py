if __name__ == '__main__':
    import argparse
    import os
    import base64
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
    from groq import Groq
    import requests

    load_dotenv()
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    parser = argparse.ArgumentParser(prog='docsum', description='summarize the input document or image')
    parser.add_argument('filename')          
    args = parser.parse_args()
    print('filename =', args.filename)

    def llm(text):
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content

    def split_text(text, max_chunk_size=1000):
        '''
        >>> split_text('abcdefg', max_chunk_size=2)
        ['ab', 'cd', 'ef', 'g']
        '''
        accumulator = []
        while len(text) > 0:
            accumulator.append(text[:max_chunk_size])
            text = text[max_chunk_size:]
        return accumulator

    def summarize_text(text):
        '''
        our current problem is we cannot summarize large documents

        you can split the document and then summarize those chunks. Then call sumarize_text to those documents
        '''
        prompt = f'''
        summarize the following text in 1-3 sentences in english.

        {text}
        '''
        try:
            output = llm(prompt)
            return output.split('\n')[-1]
        except Exception as e:
            chunks = split_text(text, 10000)
            summaries = []
            for i, chunk in enumerate(chunks):
                print('i =', i)
                summaries.append(summarize_text(chunk))
            return summarize_text(' '.join(summaries))

    def llm_image_url(url):
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {'type': 'text', 'text': "What is in this image?"},
                        {'type': 'image_url', 'image_url': {'url': url}}
                    ]
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
        )
        return completion.choices[0].message.content

    def llm_image_local(image_path):
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        return chat_completion.choices[0].message.content

    filename = args.filename

    try:
        if filename.startswith("http"):
            if any(ext in filename.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
                result = llm_image_url(filename)
            else:
                response = requests.get(filename)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    result = summarize_text(text)

        elif filename.endswith(".txt"):
            with open(filename, 'r', encoding='utf-8') as f:
                result = summarize_text(f.read())

        elif filename.endswith(".html"):
            with open(filename, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), features='html.parser')
                result = summarize_text(soup.get_text())

        elif any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]):
            result = llm_image_local(filename)

        elif filename.endswith('.pdf'):
            import fitz  
            doc = fitz.open(filename)
            text = ''
            for page in doc:
                text += page.get_text()
            result = summarize_text(text)
        
        else:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    print("📄 Treating as plain text (no extension)...")
                    result = summarize_text(f.read())
            except Exception as e:
                print(e)

        print('Summary=', result)
   
    except Exception as e:
        print(e)


       