📄 PDF Q&A with LangChain, Hugging Face, and Streamlit

This app allows you to upload PDFs and ask questions about them using state-of-the-art language models (like Mistral-7B or Flan-T5) with LangChain, FAISS, and Streamlit.

🚀 Features

📚 Upload multiple PDFs

🔍 Ask questions and get accurate answers based on your documents

🤖 Choose between lightweight or powerful LLMs (e.g., Flan-T5 or Mistral-7B)

🧠 Uses vector search (FAISS) + embeddings (BGE) for semantic matching

🧼 Simple, clean UI built with Streamlit

🧰 Tech Stack

Streamlit
 – UI

LangChain
 – Framework

Hugging Face Transformers
 – LLMs

FAISS
 – Vector search

Sentence Transformers
 – Embeddings (bge-small-en-v1.5)

📦 Installation

Clone the repo:

git clone https://github.com/yourusername/pdf-qa-app.git
cd pdf-qa-app


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Set up environment variables:

Create a .env file with your Hugging Face token:

HUGGINGFACEHUB_API_TOKEN=your_token_here


You can get your token here: https://huggingface.co/settings/tokens

🏃‍♂️ Run Locally

Start the Streamlit app:

streamlit run app.py


Then open your browser to: http://localhost:8501

☁️ Deploy to Streamlit Cloud

Push your code to GitHub.

Go to Streamlit Cloud
 and click "New app".

Connect your GitHub repo.

In the app settings:

Repository: your-username/pdf-qa-app

Branch: main (or whichever you're using)

Main file: app.py

Add your Hugging Face API token:

Go to “Advanced Settings” > “Secrets”

Add a new secret:

Name: HUGGINGFACEHUB_API_TOKEN

Value: your actual token

Click "Deploy" — your app will be live in seconds! 🎉

🧪 Requirements
requirements.txt
streamlit
langchain
langchain-community
sentence-transformers
faiss-cpu
python-dotenv
transformers>=4.41.0

📁 File Structure
pdf-qa-app/
│
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .env                   # Your Hugging Face token (DO NOT COMMIT THIS)
└── README.md              # You're reading it!

📸 Screenshots
<details> <summary>📄 PDF Upload</summary>

</details> <details> <summary>💬 Ask Questions</summary>

</details>
⚠️ Notes

mistralai/Mistral-7B-v0.1 is a large model — only use it if you’re on GPU or a paid Hugging Face inference endpoint.

google/flan-t5-base is fast and runs well on CPUs (ideal for Streamlit Cloud).

🤝 License

MIT — free to use, clone, or modify
