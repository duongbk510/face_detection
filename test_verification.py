import cv2
import sys
import numpy as np
import datetime
import os
import glob
from face_detect.RetinaFace.retinaface import RetinaFace
import time
from skimage import transform as trans
from face_embedding.arcface import Arcface
from scipy.spatial.distance import cosine
import math 
from scipy.spatial import distance as dist

from skimage import io
from skimage import img_as_ubyte
import face_alignment


# config params
thresh_detect_face = 0.8
scales = [1024, 1980]
count = 1
gpuid = -1 # use 0 for deploy server
path_model_detect_face = '/Users/pmkh/Documents/cmnd-project/face-recognition/face_detection/insightface/RetinaFace/models/retinaface-R50/R50'
path_model_embedding_face = '/Users/pmkh/Documents/cmnd-project/face-recognition/insightface/models/model-r100-ii/model,0'
image_size_embedding_face = '112,112'
# load models
model_detector = RetinaFace(path_model_detect_face, 0, gpuid, 'net3')
model_embedding = Arcface.ArcfaceModel(gpu=gpuid, model= path_model_embedding_face, image_size=image_size_embedding_face)
if gpuid >=0:
    model_detector_fanface = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
else:
    model_detector_fanface = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

def get_face(img, scales = scales):
    # cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if im_size_max < target_size:
        im_scale = 1.0
    else:
        im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    flip = False
    # for c in range(count):
    faces, landmarks = model_detector.detect(img, thresh_detect_face, scales=scales, do_flip=flip)
    # print(c, faces.shape, landmarks.shape)
    return faces, landmarks

def face_alignmet(cv_img, dst, dst_w, dst_h):
    if dst_w == 96 and dst_h == 112:
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32)
    elif dst_w == 112 and dst_h == 112:
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32)
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

def get_face_aligned(image):
    check = 1
    index_max = 0
    dist_max = 0
    cv_img = image.copy()
    # cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    faces, landmarks = get_face(image)
    if faces is not None:
        # print('find face')
        # dist_max = 0
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            if dist.euclidean((box[0], box[1]), (box[2], box[3])) > dist_max:
                dist_max = dist.euclidean((box[0], box[1]), (box[2], box[3]))
                index_max = i
        if landmarks is not None:
            # print('find lanmarks')
            landmark5 = landmarks[index_max].astype(np.int)
            face_align = face_alignmet(cv_img, landmark5, 112, 112)
            return face_align, check
        else:
            check = 0
            return image, check
    else:
        check = 0
        print('detect failed')
        return image, check

def get_face_aligned_fanface(image_path, scales=[720,1280]):
    check = 1
    img = cv2.imread(image_path)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if im_size_max < target_size:
        im_scale = 1.0
    else:
        im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
    scales = [im_scale]
    img_scale = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    sk_img = cv2.cvtColor(img_scale, cv2.COLOR_BGR2RGB)
    ldmk = model_detector_fanface.get_landmarks(sk_img)
    if ldmk is None:
        check = 0
        return img, check
    else:
        points = ldmk[0]
        p1 = np.mean(points[36:42,:], axis=0)
        p2 = np.mean(points[42:48,:], axis=0)
        p3 = points[33,:]
        p4 = points[48,:]
        p5 = points[54,:]
        dst = np.array([p1,p2,p3,p4,p5],dtype=np.float32)
        face_112x112 = face_alignmet(img_scale, dst, 112, 112)
        return face_112x112, check
        # return img, check

def get_embedding(face_align, required_size=(112, 112)):
    # face_align = cv2.resize(face_align, required_size)
    face_align = cv2.cvtColor(face_align, cv2.COLOR_BGR2RGB)
    face_align = np.transpose(face_align, (2, 0, 1))
    embedding = model_embedding.get_feature(face_align)
    return embedding

def face_distance_to_conf(face1, face2, face_match_threshold=0.7):
    face_distance = cosine(face1, face2)
    # face_distance = np.linalg.norm(face1-face2)
    # print(face_distance)
    if face_distance < 0:
        return 0
    else:
        if face_distance > face_match_threshold:
            range = (1.0 - face_match_threshold)
            linear_val = (1.0 - face_distance) / (range * 2.0)
            return linear_val
        else:
            range = face_match_threshold
            linear_val = 1.0 - (face_distance / (range * 2.0))
            return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

if __name__ == "__main__":
    path_1 = '/Users/pmkh/Documents/cmnd-project/face-recognition/img-tron/fail/13432977_417443923.jpg'
    path_2 = '/Users/pmkh/Documents/cmnd-project/face-recognition/img-tron/fail/13432977_417445330.jpg'
    # image_1 = cv2.imread(path_1)
    # image_2 = cv2.imread(path_2)
    # face_align_1, _ = get_face_aligned(image_1)
    # face_align_2, _ = get_face_aligned(image_2)
    face_align_1, _ = get_face_aligned_fanface(path_1)
    face_align_2, _ = get_face_aligned_fanface(path_2)
    embedding_1 = get_embedding(face_align_1)
    embedding_2 = get_embedding(face_align_2)
    # em1 = l2_normalize(embedding_1)
    # em2 = l2_normalize(embedding_2)
    # print(embedding_1)

    print(face_distance_to_conf(embedding_2, embedding_1))
    # print(math.log(np.dot(embedding_1, embedding_2.T) / np.linalg.norm(embedding_1-embedding_2)), 10)
    print('sim ', np.dot(embedding_1, embedding_2.T))
    print('find dt cosine ', findCosineDistance(embedding_1, embedding_2))
    print('distance euclid ', np.linalg.norm(embedding_1 - embedding_2))
    print('distance cosine ', cosine(embedding_1, embedding_2))
    # print('distance euclid 2 ', dist.euclidean(em1, em2))

    cv2.imshow('f1', face_align_1)
    cv2.imshow('f2', face_align_2)
    cv2.waitKey(0)
