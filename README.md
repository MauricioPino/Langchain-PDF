# langchain-pdf

#How its works
This app receive a PDF file. That PDF will divide it and save it in chunks or vectors. Using OpenAi Embeddings. Then, when you ask something about the PDF, the app will search related information with your question in the chunks,
previously created. Finally will return the response, should be the answer to your question.

* Install requierements
 - pip install -r requirements.txt
 - You will also need to add your OpenAI API key to the `.env` file.
 - streamlit run app.py
