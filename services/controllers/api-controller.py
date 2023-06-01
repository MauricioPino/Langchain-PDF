from flask import Flask, request
from privateGPT import PrivateGPT

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
    gpt = PrivateGPT()

    # Refactor this part... include..
    image = gpt.generate_image_from_text(text)

    # Return the image and 200 OK.
    return image, 200

if __name__ == '__main__':
    app.run()
