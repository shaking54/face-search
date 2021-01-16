import numpy as np
import cv2
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
import os
import model


path = open('dir.txt', 'r').read()

image_dirpath = 'static\img'
image_size = 160
model_path = path

model = model.InceptionResNetV1(weights_path=model_path)

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def align_images(filepath,margin):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    aligned_images = []
    img = imread(filepath)
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for (x,y,w,h) in faces:
      cropped = img[y-margin//2:y+h+margin//2,
                      x-margin//2:x+w+margin//2, :]
      aligned = resize(cropped, (image_size, image_size), mode='reflect')
      aligned_images.append(aligned)
    return np.array(aligned_images)

def features_extraction(filepath, margin=10,batch_size=1):
    aligned_images = prewhiten(align_images(filepath, margin))
    #print(len(aligned_images))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

def check_register(face, filename):
  global database
  if database is None:
      database.append(face)
  for f in database:
      if distance.euclidean(face[0], f[1]) < 0.6:
        f[2].append(filename)
        return database
  new_face = ("FaceID"+str(len(database)+1), face[0], [filename])
  database.append(new_face)

string = '00000'
def count_digit(num):
  count = 0
  while num >= 1:
      count+=1
      num/=10
  return count

a = 10001
string[:0]
print("FaceID"+string[:5-count_digit(a)]+str(a))

def check_register2(face,filename):
  global database
  new_face = ("FaceID"+string[:5-count_digit(len(database))]+str(len(database)), face, [filename])
  database.append(new_face)

def search(face):
  for f in database:
    print(distance.euclidean(face[0], f[1]))
    if distance.euclidean(face[0], f[1]) < 0.7:
        return f[2]

def search2(face, dis="euclidean"):
  scoreboard = []
  if dis == "euclidean":
    for f in database:
      print(distance.euclidean(face[0], f[1]))
      scoreboard.append((f[2], distance.euclidean(face[0], f[1])))
      # if distance.euclidean(face[0], f[1]) < 0.7:
      #     return f[2]
    return scoreboard
  if dis == "cosine":
    for f in database:
      print(distance.cosine(face[0], f[1]))
      scoreboard.append((f[2], distance.cosine(face[0], f[1])))
      # if distance.euclidean(face[0], f[1]) < 0.7:
      #     return f[2]
    return scoreboard

def search3(face, database, dis='euclidean'):
  scoreboard = []
  if dis == "euclidean":
    for f in database:
      #print(distance.euclidean(face[0], f['features vector']))
      scoreboard.append((f['filename'], distance.euclidean(face[0], f['features vector'])))
      # if distance.euclidean(face[0], f[1]) < 0.7:
      #     return f[2]
    return scoreboard
  if dis == "cosine":
    for f in database:
      #print(distance.cosine(face[0], f['features vector']))
      scoreboard.append((f['filename'], distance.cosine(face[0], f['features vector'])))
      # if distance.euclidean(face[0], f[1]) < 0.7:
      #     return f[2]
    return scoreboard



"""#*create database*"""

database = []
image_filepaths = []
image_dirpath="/static/img/"
path = os.listdir(image_dirpath)
path.sort()
count = 0
for f in path:
  if f.endswith(".jpg") or f.endswith(".png"):
     #count+=1
     try:
        #print(count)
        embs = features_extraction(image_dirpath + f)
        for i in range(len(embs)):
            check_register2(embs[i], f)
     except:
        print(f)
print(len(database))

temp = sorted(database, key=lambda score:score[2])

database = sorted(database, key=lambda score:score[2])

"""#Saving features to Json"""

dic = {}
import json
for i in range(len(database)):
  with open("/static/feature/{filename}.json".format(filename=database[i][0]),"w", encoding='utf-8') as f:
    json.dump({'ID': database[i][0],'features vector': database[i][1].tolist(), 'filename': "./static/img/"+database[i][2][0]}, f)

f1data = "" 
a = os.listdir('/static/img')
a.sort()
for file in a:
    with open('/static/img'+file, "r") as infile:
        f1data += infile.read()
        f1data += "\n"
with open('./features.json', "w") as f1:
        f1.write(f1data)



