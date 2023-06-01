#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import PyPDF2
import imgkit
import re

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        if contains_financial_info(query):
            generate_image_from_text(query)
        else:
            print("El texto no contiene información financiera.")
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

            # Generate image from PDF with financial information
            if document.metadata["source"].endswith(".pdf"):
                generate_image_from_pdf(document.metadata["source"])

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

def generate_image_from_pdf(file_path: str):
    output_file = file_path.replace('.pdf', '.png')
    options = {
        'quiet': '',
        'disable-smart-width': ''
    }
    imgkit.from_file(file_path, output_file, options=options)

def contains_financial_info(text: str) -> bool:
    # List of financial-related keywords
    financial_keywords = ["finance", "economy", "investment", "stocks", "dividends", "earnings", "income",
                          "balance", "accounting", "loan", "mortgage", "interest", "credit", "debt",
                          "credit card", "taxes", "tax return", "refund", "inflation",
                          "stock market", "stock exchange", "quote", "stock quote", "funds",
                          "assets", "liabilities", "cash flow"]

    # Regular expressions to search for common financial patterns
    financial_patterns = [r"\$\d+(?:\.\d+)?",  # Dollars and decimal numbers ($100, $10.50, etc.)
                          r"\d+(?:,\d+)*(?:\.\d+)?%",  # Percentages (25%, 3.5%, etc.)
                          r"\d+(?:,\d+)*(?:\.\d+)?[MB]?",  # Amounts (100, 1.5M, 10.2B, etc.)
                          r"\d{2}/\d{2}/\d{4}",  # Dates (MM/DD/YYYY)
                          r"\d{2}-\d{2}-\d{4}"]  # Dates (MM-DD-YYYY)

    # Check if the text contains any financial-related keywords
    for keyword in financial_keywords:
        if keyword.lower() in text.lower():
            return True

    # Check if the text matches any financial patterns using regular expressions
    for pattern in financial_patterns:
        if re.search(pattern, text):
            return True

    return False

from PIL import Image, ImageDraw, ImageFont

def generate_image_from_text(text: str):
    # Configuración de la imagen
    image_width = 800
    image_height = 600
    background_color = (255, 255, 255)  # Blanco
    text_color = (0, 0, 0)  # Negro
    font_size = 20
    font_path = "../langchain-pdf/fonts/OpenSans-Light.ttf"  # Ruta a tu archivo de fuente (.ttf)

    # Crear una nueva imagen con fondo blanco
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Configurar la fuente
    font = ImageFont.truetype(font_path, font_size)

    # Calcular el tamaño del texto
    text_width, text_height = draw.textsize(text, font=font)

    # Calcular la posición del texto en el centro de la imagen
    text_x = (image_width - text_width) // 2
    text_y = (image_height - text_height) // 2

    # Dibujar el texto en la imagen
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    # Guardar la imagen generada
    image.save("output_image.png")


if __name__ == "__main__":
    main()
