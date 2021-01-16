import numpy as np
import random
import cv2
from imageio import imread
from skimage.transform import resize
from scipy.spatial import distance
import model
from pyflann import *
import os

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


def align_images(filepath, margin):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    aligned_images = []
    img = imread(filepath)
    faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        cropped = img[y - margin // 2:y + h + margin // 2,
                  x - margin // 2:x + w + margin // 2, :]
        #aligned = resize(cropped, (image_size, image_size), mode=
        aligned = cv2.resize(cropped, (image_size, image_size))
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

def check_register2(face,filename):
  global database
  new_face = ("FaceID"+str(len(database)+1), face[0], [filename])
  database.append(new_face)


def search2(face, database):
    idx = []
    flann = FLANN()
    result, dists = flann.nn(
        database, face, 50, algorithm="kmeans", branching=32, iterations=7, checks=16)
    return result, dists


def search3(face, database, dis='euclidean'):
  scoreboard = []
  if dis == "euclidean":
    print(dis)
    for f in database:
      scoreboard.append((f['filename'], distance.euclidean(face[0], f['features vector'])))
    return scoreboard
  if dis == "cosine":
    print(dis)
    for f in database:
      scoreboard.append((f['filename'], 1-distance.cosine(face[0], f['features vector'])/2))
    return scoreboard

def getCropImage(faces, img, margin):
    align = []
    for (x,y,w,h) in faces:
        cropped = img[y - margin // 2:y + h + margin // 2, x - margin // 2:x + w + margin // 2, :]
        align.append(cropped)
    return align


a = os.listdir('static/img')
a = random.choices(a, k = 2000)
a.sort()
with open('query.txt', 'w') as file:
    for i in a:
        file.write(i)
        file.write('\n')




# a = features_extraction(r"E:/UIT/CS231.J21.KHTN/FinalProject/Image/Andres_Iniesta01.jpg")
# print(a)
#
# import json
# data = []
# with open('./features5.json', "r") as f:
#     for line in f:
#         #print(line)
#         data.append(json.loads(line))
#
# features = []
# for i in data:
#     features.append(i['features vector'])
# features = np.array(features)
#print(features)

#test = np.asarray([-0.018833570182323456, 0.08173629641532898, 0.14771948754787445, -0.04784661903977394, 0.1666233092546463, -0.1697635054588318, 0.08496268838644028, -0.14857304096221924, 0.10969700664281845, -0.03722261264920235, -0.06021631136536598, 0.09861565381288528, 0.07744467258453369, -0.2027350217103958, -0.0345672145485878, -0.0550990104675293, -0.08657114952802658, 0.09918823093175888, -0.02832387387752533, 0.05393557623028755, 0.0535411462187767, 0.05943223461508751, -0.05923587828874588, 0.010143148712813854, 0.023173507302999496, 0.07182755321264267, 0.033506862819194794, 0.014576930552721024, -0.08913381397724152, -0.022929273545742035, -0.017676739022135735, -0.13391731679439545, -0.07280252873897552, -0.15016677975654602, -0.002398615935817361, -0.06314869225025177, 0.07090810686349869, -0.17865325510501862, -0.03459160029888153, -0.02973913960158825, -0.13820333778858185, -0.0044710226356983185, 0.12293374538421631, 0.15036620199680328, 0.12164320796728134, 0.09778906404972076, -0.10887320339679718, 0.04958316311240196, -0.07115611433982849, 0.022430075332522392, -0.05071217194199562, -0.03188732638955116, -0.11872808635234833, -0.025391239672899246, -0.04604322463274002, -0.10225985944271088, -0.019349422305822372, -0.012215061113238335, -0.02349337935447693, 0.041876502335071564, 0.09947200119495392, 0.024428119882941246, 0.01651066541671753, -0.05258328467607498, -0.14243865013122559, 0.22828836739063263, -0.12188972532749176, -0.05001175031065941, 0.09772596508264542, -0.04236175864934921, 0.016084762290120125, -0.04102618619799614, -0.05267402157187462, -0.0029596418607980013, -0.09070280939340591, -0.058523837476968765, 0.016878092661499977, 0.017417751252651215, 0.08397309482097626, -0.0073594218119978905, -0.05164705589413643, 0.00027740656514652073, -0.0440223291516304, 0.13943511247634888, 0.054179005324840546, -0.10671231895685196, 0.08990316838026047, -0.12120332568883896, 0.008709360845386982, -3.975002618972212e-05, -0.05284940451383591, 0.02239956147968769, 0.08713909238576889, 0.02494058758020401, -0.09143570810556412, -0.005721262656152248, -0.031662847846746445, 0.004546430427581072, -0.07183007895946503, -0.09345787018537521, 0.10821665078401566, -0.03525419160723686, 0.19958792626857758, -0.11791252344846725, -0.1191452294588089, -0.0028122344519943, -0.11411554366350174, -0.059662673622369766, -0.13194623589515686, 0.014413950964808464, 0.11015959084033966, -0.09760608524084091, 0.13305379450321198, 0.0390644408762455, 0.09516675025224686, 0.2625126540660858, -0.06817318499088287, 0.13218146562576294, -0.01973126269876957, 0.14865057170391083, 0.008346897549927235, 0.03211098536849022, -0.025236306712031364, 0.05127166211605072, -0.04679165780544281, -0.0720304623246193, 0.027415260672569275, -0.09264826774597168])
# #print(test)
# test = features_extraction('E:/UIT/CS231.J21.KHTN/FinalProject/Image/Andres_Iniesta01.jpg')
# t = np.asarray(test, )
# print(t.shape)
# a,b = search2(t, features)
# print(a)

# data = []
# with open('./features.json', "r") as f:
#     for line in f:
#         data.append(json.loads(line))
