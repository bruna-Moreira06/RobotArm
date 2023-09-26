import cv2
import mediapipe as mp
import numpy as np


import os
import sys
import time


sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        ip = '192.168.1.207'
        if not ip:
            print('input error, exit')
            sys.exit(1)
########################################################


arm = XArmAPI(ip, is_radian=True)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Position initiale du robot
x0 = 200
z0 = 140
y0 = 0
arm.set_position(x=x0, y=y0, z=z0, roll=180, pitch=0, yaw=0, speed=100, is_radian=False, wait=True)
# For webcam input:
cap1 = cv2.VideoCapture(0)
time.sleep(1)
cap2 = cv2.VideoCapture(1)

# For webcam input:

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap1.isOpened():
    success, image = cap1.read()
    image = cv2.resize(image,(640,480))
    if cap2.isOpened():
        _,image2 = cap2.read()
        image2 = cv2.resize(image2,(640,480))
        x_rect = int(640*0.25)
        y_rect = int(480*0.9)
        x_rect2 = int(640*0.9)
        y_rect2 = int(480*0.1)
        cv2.rectangle(image,(x_rect,y_rect),(x_rect2,y_rect2),(0,0,255),2)
        cv2.rectangle(image2,(x_rect,y_rect),(x_rect2,y_rect2),(0,0,255),2)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    image2.flags.writeable = False
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    results2 = pose.process(image2)

    # Draw the pose annotation on the image.
    image2.flags.writeable = True
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image2,
        results2.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    if results.pose_landmarks:
        posX = int(100*results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x)
        posY = int(100*results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y)
    else:
        posX = 0
        posY = 0
        
    if results2.pose_landmarks:
        posZ = int(100*results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x)
    else:
        posZ = 0
        
    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)
    cv2.putText(image,f"x:{posX}",[80,80], cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    cv2.putText(image,f"y:{posY}",[80,120], cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    cv2.putText(image2,f"z:{posZ}",[80,80], cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    hstack = np.hstack((image,image2))
    cv2.imshow('hole',hstack)
    if cv2.waitKey(5) & 0xFF == 27:
      break
        
      
    #transform MP for robot coord
        
    '''
    Module d'initialisation de la position de départ du pilote dans l'espace
    '''
    def z_Humain(posZ):
        zR = ( 13 * posZ ) -510
        return zR
            
    xR = z_Humain(posZ)
    yR = (posX * 8) - 460
    zR = (posY * -6.5) + 595
                
    #clip to the rail
    xR=np.clip(xR,x0+10,530)
    yR=np.clip(yR,y0-260,y0+260)
    zR=np.clip(zR,z0+10,z0+530)
    # if debug == True:
        #     print(xR,yR,zR)
    arm.set_position(x=xR, y=yR, z=zR, roll=180, pitch=0, yaw=0, speed=100, is_radian=False, wait=True)
    
cap1.release()