# Implementing-RAG-with-Hybrid-Search-for-Efficient-Document-Retrieval
 involves combining Retrieval-Augmented Generation (RAG) techniques with hybrid search methods to enhance document retrieval and contextual understanding.

📄 Overview
This project leverages free-tier accounts for Google Document AI, OpenAI, and Pinecone to create a powerful NLP workflow.

🔹 OpenAI Usage Note:
Depending on usage volume, OpenAI may require a small amount (e.g., $5) in your account to facilitate API calls.

🔹 Model Recommendation:
You can try using model="text-embedding-ada-002" in line 57 (embedding client). 
However, performance may vary with this approach under free-tier usage limits.

This project is a [Type/Description, e.g., Retrieval-Augmented Generation system] built for [specific purpose use case].
It allows you to search, generate embeddings, manage free plan Pinecone index operations, and utilize vector search mechanisms in Python with integrated embeddings and models.

This project leverages BM25 embeddings, vector embeddings, retrieval-augmented search, Pinecone, and advanced NLP workflows.
✨ Features

    Sparse Encoder Training: Train embeddings on large corpora using BM25 encoder models.
    Vector Preprocessing: Seamlessly preprocess embeddings for vector stores (Pinecone).
    Vector Visualization & Preview: View embeddings using tabular representations for ease.
    Cross-platform Index Management: Create/clear indexes dynamically and efficiently.
    Custom Search Functions: Search by query and context for embeddings with similarity matching.
    Streamlined Console Operations: Clear console on key actions for a better user experience.

🛠️ Installation

Follow these steps to install the project dependencies:

    Clone the repository:

git clone https://github.com/Fabiqula/Implementing-RAG-with-Hybrid-Search-for-Efficient-Document-Retrieval.git
cd Implementing-RAG-with-Hybrid-Search-for-Efficient-Document-Retrieval

    Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

    Install the required dependencies:

pip install -r requirements.txt

🖥️ Usage Instructions

Once installed, execute the main script with:

python main.py
After setting up the credentials and ensuring that defaulted Churchill speach pdf works,
you can upload your own pdf file at the top in the line file_path just below the imports.

Input

You can input queries using the prompt for context-keyword search. Example on our Churchill speach default pdf:

Type what are you looking for: "resistance in WWII, beaches, fight strategy"

🛠️ Dependencies

Include all the required dependencies here. Example:

    os
    bm25
    pinecone-client
    pandas
    numpy
    textwrap
    tabulate
    langchain

🏗️ How it Works
Sparse Encoding:

The encoding system leverages BM25 for embeddings creation across text inputs to allow vectorization.
Vector Preprocessing:

Vectors are reshaped and prepared for Pinecone indexing, allowing semantic search.
🐛 Development Notes

    Designed with cross-platform compatibility in mind.

    Ensure API Keys are stored securely using dotenv.
    Example:

    os.getenv("YOUR_API_KEY")

    Tested primarily on Windows/Linux.

🔗 License

This repository is licensed under the MIT License.
💬 Contact Information

If you have questions, suggestions, or want to contribute, feel free to contact me:

    GitHub: Fabiqula
    Email: gregorykowalczyk25@gmail.com

