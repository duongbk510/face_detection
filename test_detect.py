import cv2
import sys
import numpy as np
import datetime
import os
import glob
from face_detect.RetinaFace.retinaface import RetinaFace
import time
from skimage import transform as trans
from scipy.spatial import distance as dist

thresh = 0.8
scales = [1024, 1980]

count = 1

gpuid = -1
detector = RetinaFace('/Users/pmkh/Documents/cmnd-project/face-recognition/face_detection/insightface/RetinaFace/models/retinaface-R50/R50', 0, gpuid, 'net3')

img = cv2.imread('/Users/pmkh/Documents/cmnd-project/face-recognition/img-tron/IMG_20200311_164049_520.jpg')
cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
print(img.shape)
im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
# print(im_size_min, im_size_max)
if im_size_max < target_size: 
  im_scale = 1.0
else:
# if im_size_min>target_size or im_size_max>max_size:
  im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
  if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)

print('im_scale', im_scale)

scales = [im_scale]
flip = False

start = time.time()
for c in range(count):
  faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
  print(c, faces.shape, landmarks.shape)
end = time.time()
print('time: ', str(end-start))

def alignment(cv_img, dst, dst_w, dst_h):
    if dst_w == 96 and dst_h == 112:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32)
    elif dst_w == 112 and dst_h == 112:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32)
    elif dst_w == 150 and dst_h == 150:
        src = np.array([
            [51.287415, 69.23612],
            [98.48009, 68.97509],
            [75.03375, 96.075806],
            [55.646385, 123.7038],
            [94.72754, 123.48763]], dtype=np.float32)
    elif dst_w == 160 and dst_h == 160:
        src = np.array([
            [54.706573, 73.85186],
            [105.045425, 73.573425],
            [80.036, 102.48086],
            [59.356144, 131.95071],
            [101.04271, 131.72014]], dtype=np.float32)
    elif dst_w == 224 and dst_h == 224:
        src = np.array([
            [76.589195, 103.3926],
            [147.0636, 103.0028],
            [112.0504, 143.4732],
            [83.098595, 184.731],
            [141.4598, 184.4082]], dtype=np.float32)
    else:
        return None
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    face_img = cv2.warpAffine(cv_img,M,(dst_w,dst_h), borderValue = 0.0)
    return face_img

# def get_face_align(image):
#     check = 1
#     landmarks = fa.get_landmarks(image)
#     if landmarks is None:
#         # for sigma in np.linspace(0.0, 3.0, num=11).tolist():
#         #     seq = iaa.GaussianBlur(sigma)
#         #     image_aug = seq.augment_image(image)
#         #     landmarks = fa.get_landmarks(image_aug)
#         check = 0
#         return image, check
#     else:
#         points = landmarks[0]
#         p1 = np.mean(points[36:42,:], axis=0)
#         p2 = np.mean(points[42:48,:], axis=0)
#         p3 = points[33,:]
#         p4 = points[48,:]
#         p5 = points[54,:]
#         dst = np.array([p1,p2,p3,p4,p5],dtype=np.float32)
#         # print(dst)
#         cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         face_112x112 = alignment(cv_img, dst, 112, 112)
#         return face_112x112, check

print(landmarks)
print(len(landmarks))
print(faces)
if faces is not None:
  dist_max = 0
  print('find', faces.shape[0], 'faces')
  for i in range(faces.shape[0]):
    #print('score', faces[i][4])
    box = faces[i].astype(np.int)
    #color = (255,0,0)
    # find big face in image
    if dist.euclidean((box[0], box[1]), (box[2], box[3])) > dist_max:
      dist_max = dist.euclidean((box[0], box[1]), (box[2], box[3]))
      index_max = i 
    color = (0,0,255)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    if landmarks is not None:
      landmark5 = landmarks[i].astype(np.int)
    #   print(landmark5)
      face_align = alignment(cv_img, landmark5, 112, 112)
      for l in range(landmark5.shape[0]):
        color = (0,0,255)
        if l==0 or l==3:
          color = (0,255,0)
        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

  # filename = './detector_test.jpg'
  # print('writing', filename)
  # cv2.imwrite(filename, img)
  cv2.imshow('face_lanmarks', img)
  cv2.imshow('face_align', face_align)
  cv2.waitKey(0)

