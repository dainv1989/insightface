import cv2
import numpy as np
import time

from scipy.spatial import distance as dist
from imutils import face_utils
import dlib

from retinaface_cov import RetinaFaceCoV
import argparse


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help="input image file")
ap.add_argument('-o', '--output', help="output image file")
args = vars(ap.parse_args())

epsilon = 1e-3
eye_thresh = 0.3
thresh = 0.8
mask_thresh = 0.2
scales = [640, 1080]
gpuid = 0
colors = [(255, 0, 0),  # right eye
          (0, 255, 0),  # left eye
          (0, 0, 255),  # nose
          (255, 0, 255),  # right corner of mouth
          (0, 255, 255)]  # left corner of mouth
font = cv2.FONT_HERSHEY_SIMPLEX

detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

img = cv2.imread(args['input'])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
# im_scale = 1.0
# if im_size_min>target_size or im_size_max>max_size:
im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)

print('im_scale', im_scale)

scales = [im_scale]
flip = False
st = time.time()

faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

if faces is None:
    print('Found no face')
    exit(0)

# print('find', faces.shape[0], 'faces')
for i in range(faces.shape[0]):
    # print('score', faces[i][4])
    face = faces[i]
    box = face[0:4].astype(np.int)
    mask = face[5]
    # print(i, box, mask)
    # color = (255,0,0)
    if mask >= mask_thresh:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

    r_eye, l_eye, nose, r_mou, l_mou = landmarks[i]
    h_ratio = (nose[0] - r_eye[0]) / (l_eye[0] - nose[0] + epsilon)
    rotate = 'OK'
    if h_ratio < 0.6 or (nose[0] - r_eye[0]) < 0:
        rotate = 'R'
    if h_ratio > 1.5 or (l_eye[0] - nose[0]) < 0:
        rotate = 'L'
    cv2.putText(img, "{:.2f} {}".format(h_ratio, rotate), (box[0], box[3] + 15), font, 0.5, (0, 0, 255), 2)

    if rotate != '':
        # eye ratio calculation
        rect = dlib.rectangle(box[0], box[1], box[2], box[3])
        fc_shape = predictor(gray, rect)
        fc_shape = face_utils.shape_to_np(fc_shape)
        # print(fc_shape)
        leftEye = fc_shape[lStart:lEnd]
        rightEye = fc_shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        print("eye ratio LR: {:.3f} {:.3f}".format(leftEAR, rightEAR))

        # draw eye contour
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

    for j, p in enumerate(landmarks[i]):
        x, y = p.astype(np.int)
        cv2.circle(img, (x, y), 1, colors[j], 2)

if args['output'] is not None:
    print('writing', args['output'])
    cv2.imwrite(args['output'], img)

et = time.time()
print("elapsed time {:.2f}".format((et - st) * 1000))

cv2.imshow('result', img)
cv2.waitKey(0)
