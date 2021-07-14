import cv2
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

capture = cv2.VideoCapture('./videos/skipping.mp4')

p_time = 0
while True:

    _, img = capture.read()
    img = cv2.resize(img, (700, 1000), cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_img)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)

            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    c_time = time.time()
    fps = int(1/(c_time - p_time))
    p_time = c_time
    cv2.putText(img, str(fps), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('img', img)
    cv2.waitKey(1)

