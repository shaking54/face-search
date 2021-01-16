# face-search

this repos using opencv haar-cascade-classifier to detect face in images and using facenet 
to extract features

download weight of FaceNet in this [repos](https://github.com/nyoki-mtl/keras-facenet):

download my images dataset [here](https://drive.google.com/file/d/1heCra87tdRwb4yTbgY4svFjhruj_35Xk/view?usp=sharing):


Download images and weights. Set up weights dir-path in `dir.txt`. 
Note:

put image to .\static\img\ 


1.Install packages: 

``
pip install -r requirements.txt
``

2.Create features database:

``
python .\feature_extractor.py
``


3.run 

``
python .\server.py
``

4.Access 127.0.0.1:5000 to search engine
``
``

reference:


https://github.com/nyoki-mtl/keras-facenet
https://github.com/matsui528/sis
