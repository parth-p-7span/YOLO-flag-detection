from flask import jsonify, Flask, request
import os
from pathlib import Path
from yolov5.my_pred import run

root = os.getcwd()
weight_path = os.path.join(root, 'weights', 'best.pt')
UPLOAD_FOLDER = os.path.join(root, "temp")
ALLOWED_EXTENSIONS = {'jpg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/file-upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = "temp.jpg"
        temp = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(temp)
        # function call #

        data = run(weights=Path(weight_path), source=Path(temp))
        resp = jsonify({'message': 'File successfully uploaded', "FLAG COLOR": data["FLAG COLOR"],
                        "dominant color is": data["dominant color is"]})
        resp.status_code = 201
        return resp

    else:
        resp = jsonify({'message': 'Allowed only jpg'})
        resp.status_code = 400
        return resp


# run(weights=Path(weight_path), source=Path(temp))


if __name__ == "__main__":
    app.run(debug=True)
