File structure
chatbot_rag_personalized/
├── app.py                  ← Main Streamlit app
├── requirements.txt
├── config.py               ← (Optional) Store API key
├── utils/                  ← Helper modules
│   ├── chroma_handler.py
│   ├── embedding.py
│   └── personalization.py
└── uploaded_docs/          ← Uploaded PDFs/texts (optional)
