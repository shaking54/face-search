import numpy as np
from flask import Flask, request, render_template
import json

from numpy import float64

import Ultis
import cv2
from skimage.transform import resize
from imageio import imread
from PIL import Image


app = Flask(__name__)



# Read image features
data = []
with open('./features.json', "r") as f:
    for line in f:
        #print(line)
        data.append(json.loads(line))

features = []
for i in data:
    features.append(i['features vector'])
features = np.asarray(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        #img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "./static/uploaded/"+file.filename
        file.save(uploaded_img_path)

        # Run search
        query = Ultis.features_extraction(uploaded_img_path)
        result, dist = Ultis.search2(np.asarray(query,dtype=float64), features)

        scores2 = []
        print(dist)
        for i in range(len(result[0])):
            scores2.append((data[result[0][i]]['filename'],dist[0][i]))
        #print(scores2[:50])cl
        #scores2 = Ultis.search3(query, data, dis="cosine")
        #scores2 = sorted(scores2, key=lambda score: score[1], reverse=True)
        #print(scores2)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=(scores2[:50]))
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("127.0.0.1", debug=True)
