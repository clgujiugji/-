import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from stgcn.stgcn import STGCN

# 定义关键点
# ... [保持原有定义不变] ...
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

KEY_JOINTS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

POSE_CONNECTIONS = [(6, 4), (4, 2), (2, 13), (13, 1), (5, 3), (3, 1), (12, 10),
                    (10, 8), (8, 2), (11, 9), (9, 7), (7, 1), (13, 0)]

POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck

LINE_COLORS = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222),
               (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255),
               (255, 156, 127), (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

ACTION_MODEL_MAX_FRAMES = 30

class FallDetection:
    def __init__(self):
        self.action_model = STGCN(weight_file='./weights/tsstg-model.pth', device='cpu')
        self.joints_list = deque(maxlen=ACTION_MODEL_MAX_FRAMES)

    def detect(self):
        cap = cv2.VideoCapture(0)
        image_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        image_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        with mp_pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if not results.pose_landmarks:
                    continue

                landmarks = results.pose_landmarks.landmark
                joints = np.array([[landmarks[joint].x * image_w,
                                    landmarks[joint].y * image_h,
                                    landmarks[joint].visibility]
                                   for joint in KEY_JOINTS])
                self.joints_list.append(joints)

                if len(self.joints_list) == ACTION_MODEL_MAX_FRAMES:
                    pts = np.array(self.joints_list, dtype=np.float32)
                    out = self.action_model.predict(pts, (image_w, image_h))
                    action_name = self.action_model.class_names[out[0].argmax()]

                    if action_name == 'Fall Down':
                        print("摔倒")
                        #break  # 检测到摔倒后退出循环

        cap.release()

if __name__ == '__main__':
    FallDetection().detect()
