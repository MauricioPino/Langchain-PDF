from flask import Flask, jsonify, request
import privateGPT
import ingest
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/generate-image-by-text', methods=['POST'])
def generate_image():
    # Verify that request has a file.
    if 'file' not in request.files:
        return 'The file is null', 400

    file = request.files['file']

    # Checking if the file has a valid extension 
    if file.filename == '':
        return 'The file is null', 400
    if not file.filename.endswith('.txt'):
        return 'The file must has a valid extension .txt', 400

    # Reading the content..
    text = file.read().decode('utf-8')

    # Instance
    gpt = privateGPT

    # Refactor this part... include..
    image = gpt.generate_image_from_text(text)

    # Return the image and 200 OK.
    return image, 200

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "File is null"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Filename is empty"})

    filename = secure_filename(file.filename)
    file.save(f"source_documents/{filename}")

    ing = ingest
    ing.main()

    return jsonify({"message": "File was processed and the model was trained succesfully!"}, 200)


@app.route("/ask-question", methods=["POST"])
def ask_question():

    gpt = privateGPT
    gpt.main

if __name__ == '__main__':
    app.run()
