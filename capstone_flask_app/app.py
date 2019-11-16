from __future__ import division, print_function
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, render_template
import numpy as np
import os


app = Flask(__name__)

m_path = '/Users/vitale/Desktop/GW_data_science/GW_2018-2019/Fall_2019/machine_learning_2/capstone_flask_app/plant_classifier_1.hdf5'
model = load_model(m_path)

def predict(img):
    im = image.load_img(img, target_size=(150, 150), interpolation="lanczos")
    im = image.img_to_array(im)
    im = im / 255
    im = np.expand_dims(im, axis=0)
    result = model.predict(im)
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print("predicting!!!!!")
        image_file = request.files['image']
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, 'images', secure_filename(image_file.filename))
        image_file.save(file_path)
        prediction = predict(file_path)
        class_labels = {0: 'Asclepias tuberosa', 1: 'Cercis canadensis', 2: 'Cichorium intybus', 3: 'Cirsium vulgare',
                        4: 'Claytonia virginica', 5: 'Gaillardia pulchella', 6: 'Glechoma hederacea',
                        7: 'Liquidambar styraciflua', 8: 'Lonicera japonica', 9: 'Lotus corniculatus',
                        10: 'Parthenocissus quinquefolia', 11: 'Phytolacca americana', 12: 'Prunella vulgaris',
                        13: 'Prunus serotina', 14: 'Rosa multiflora', 15: 'Rudbeckia hirta', 16: 'Taraxacum officinale',
                        17: 'Trifolium pratense', 18: 'Verbascum thapsus', 19: 'Viola sororia'}
        pred = prediction.argmax(axis=-1)
        result = class_labels[pred[0]]
        return result
    return None

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()


# run with:
# cd /Users/vitale/Desktop/GW_data_science/GW_2018-2019/Fall_2019/machine_learning_2/capstone_flask_app
# export FLASK_APP=app.py
# flask run