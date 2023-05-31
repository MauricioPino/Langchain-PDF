import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import fitz

pdf_directory = "pdf_folder"
embeddings_model_name = "all-MiniLM-L6-v2"
persist_directory = "persist_directory"

def main():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Upload documents from folder
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            document = load_pdf(file_path)
            documents.append(document)

    # Divide the elements in chunks
    chunks = db.text_splitter.split_documents(documents)

    llm = GPT4All(model="gpt2", n_ctx=1024, backend='gptj', callbacks=[StreamingStdOutCallbackHandler()])

    retriever = db.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    while True:
        query = input("Ask a question, or write 'exit': ")
        if query == "exit":
            break

        res = qa(query, chunks)
        answer = res['result']

        print("\n> Ask the question:")
        print(query)
        print("\n> Answer:")
        print(answer)

def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    return text

if __name__ == "__main__":
    main()


