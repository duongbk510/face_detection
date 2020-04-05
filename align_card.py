import cv2
import sys
import numpy as np
from face_detect.RetinaFace.retinaface import RetinaFace
from scipy.spatial import distance as dist
import imutils
from math import atan, degrees

# config params for detection
thresh = 0.8
scales = [1024, 1980]
count = 1
gpuid = -1
model_detector = RetinaFace('/home/duongnh/passport-ocr/face_detect/RetinaFace/models/retinaface-R50/R50', 0, gpuid, 'net3')

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
    faces, landmarks = model_detector.detect(img, thresh, scales=scales, do_flip=flip)
    # print(c, faces.shape, landmarks.shape)
    # return faces, landmarks
    index_max = 0
    dist_max = 0
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
            return landmark5
        else:
            return None
    else:
        return None

def align_image(img):
    alpha = 0
    landmarks = get_face(img)
    if landmarks is None:
        return img
    else:
        eye_left = landmarks[0]
        eye_right = landmarks[1]
        nose = landmarks[2]
        # print(landmarks)
        cv2.circle(img, (eye_left[0], eye_left[1]), 1, (0,0,255), 2)
        cv2.circle(img, (eye_right[0], eye_right[1]), 1, (0,0,255), 2)
        cv2.circle(img, (nose[0], nose[1]), 1, (0,0,255), 2)
        # rotation image by 2 eyes
        canh_doi = abs(eye_left[1] - eye_right[1])
        canh_ke = abs(eye_left[0] - eye_right[0])
        if canh_doi == 0:
            return img
        elif canh_ke == 0 and eye_left[1] < eye_right[1]:
            alpha = 90
            rotated_image = imutils.rotate_bound(img, alpha)
            return rotated_image
        elif canh_ke == 0 and eye_left[1] > eye_right[1]:
            alpha = 270
            rotated_image = imutils.rotate_bound(img, alpha)
            return rotated_image
        else:
            alpha = degrees(atan(canh_doi / canh_ke))
            print(alpha)
            # 4 cases loop for top, bottom, left, right
            if eye_left[0] < eye_right[0] and eye_left[1] < eye_right[1]:
                print('case 1')
                alpha = 360 - alpha
                rotated_image = imutils.rotate_bound(img, alpha)
                return rotated_image
            elif eye_left[0] < eye_right[0] and eye_left[1] > eye_right[1]:
                print('case 2')
                alpha = alpha + 0
                rotated_image = imutils.rotate_bound(img, alpha)
                return rotated_image
            elif eye_left[0] > eye_right[0] and eye_left[1] > eye_right[1]:
                print('case 3')
                alpha = 180 - alpha
                rotated_image = imutils.rotate_bound(img, alpha)
                return rotated_image
            else: # eye_left[0] > eye_right[0] and eye_left[1] < eye_right[1]
                print('case 4')
                alpha = 180 + alpha 
                print(alpha)
                rotated_image = imutils.rotate_bound(img, alpha)
                return rotated_image 
        

if __name__ == "__main__":
    img = cv2.imread('/home/duongnh/passport-ocr/test_img/235560369_454429277.jpg')
    # landmarks = get_face(img = img)
    # print(landmarks)
    rotated_image = align_image(img)
    # rotated_image = imutils.rotate(img, 90)
    cv2.imshow('result', rotated_image)
    cv2.waitKey(0)