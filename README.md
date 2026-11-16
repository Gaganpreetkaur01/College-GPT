🎓 College GPT – AI-Powered College Assistant

College GPT is an AI-powered college assistant designed to help students get answers to college-related and general questions using Gemini AI and Retrieval-Augmented Generation (RAG).

It allows students to ask questions about academic documents, syllabi, fees, university rules and other college information.

🚀 Features

- AI Chatbot: Answers general questions using Gemini AI.
- PDF-Based Question Answering: Reads and searches information from uploaded college PDF documents.
- RAG Architecture: Retrieves relevant information from college documents before generating answers.
- Conversation History: Maintains previous messages for contextual follow-up questions.
- PDF References: Displays source PDF names and page numbers with answers.
- Multilingual Support: Supports English, Hindi and Punjabi.
- Chat Interface: Interactive ChatGPT-style interface built with Streamlit.
- Automatic PDF Indexing: Processes PDF documents and stores searchable information.

🛠️ Technologies Used

- Python
- Streamlit
- Google Gemini API
- ChromaDB
- Google GenAI SDK
- PyPDF
- Tiktoken

⚙️ How It Works

1. Students upload or place college PDF documents in the PDFs folder.
2. The application extracts text from the PDF documents.
3. Extracted text is divided into smaller chunks.
4. Gemini Embedding converts text into vector embeddings.
5. ChromaDB stores the embeddings and document information.
6. When a student asks a question, relevant information is retrieved from the database.
7. Gemini AI generates an answer using the retrieved context.

📂 Project Structure

College-GPT/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── PDFs/
│   └── College PDF documents
│
└── college_database/
    └── Automatically generated database

Note: PDFs and college_database are excluded from the GitHub repository.

💻 Installation and Setup

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/College-GPT.git

2. Open the project folder

cd College-GPT

3. Create a virtual environment

python -m venv .venv

4. Activate the environment (Windows)

.venv\Scripts\activate

5. Install dependencies

pip install -r requirements.txt

6. Configure Gemini API Key

Set your Gemini API key in PowerShell:

$env:GEMINI_API_KEY="YOUR_API_KEY"

Replace "YOUR_API_KEY" with your actual Gemini API key.

Never share or upload your API key publicly.

7. Run the application

streamlit run app.py

The application will open in your browser.

🔑 API Requirements

This project requires a Google Gemini API key.

Get your API key from:

https://aistudio.google.com/apikey

🎯 Project Objective

The main objective of College GPT is to make college information easily accessible to students through an intelligent conversational assistant.

It combines Generative AI, vector databases and document retrieval to provide a more useful and interactive learning experience.

👩‍💻 Developed By

Gaganpreet Kaur

B.Tech – Computer Science and Engineering

Project: College GPT