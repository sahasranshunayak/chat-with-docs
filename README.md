# Chat With Your Documents

I built this to let you upload a PDF or CSV and just ask it questions instead of reading through the whole thing. You paste your own free Groq API key, upload a file, and start asking.

Live demo:  tesseract_cmd
## How it works

The document gets split into small chunks of text. Each chunk gets turned into a vector and stored in a local database (Chroma). When you ask something, it finds the chunks that actually relate to your question and sends just those to the AI model, instead of dumping the whole document in. That's basically what "RAG" (Retrieval-Augmented Generation) means, it's the same idea used in tools like ChatGPT's file upload feature.

## Built with

- Streamlit for the interface
- Groq's API for the AI model (it's free and fast)
- ChromaDB to store and search the document chunks
- pypdf and pandas for reading the files

## Running it yourself

```bash
python -m venv venv
venv\Scripts\activate     # on Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Groq API key from console.groq.com/keys, then paste it into the sidebar once the app opens.

## Why I made this

I'm a first-year CS student, still learning most of this as I go. I wanted something real to build instead of just watching tutorials, so I made this end to end, wrote the code, fixed the bugs, deployed it, all of it. It's also the base project I use for freelance work involving AI chatbots and document tools.

## Things I might add later

- Handling multiple files at once
- Showing which part of the document an answer came from
- A summarize button
