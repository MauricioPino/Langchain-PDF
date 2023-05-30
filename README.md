# langchain-pdf

This repository contains an app that works with PDF files. The app divides the PDF into smaller chunks or vectors using OpenAI Embeddings. When you ask a question about the PDF, the app searches for related information within the preprocessed chunks. Finally, it returns a response that should be the answer to your question.

## How it Works
1. Install the required dependencies:
2. Add your OpenAI API key to the `.env` file.
3. Run the following command in your terminal to start the app: streamlit run app.py

Make sure to have the necessary dependencies and configurations in place to run the app successfully.
