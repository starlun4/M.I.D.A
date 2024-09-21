import kivy
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.graphics import Color, Rectangle
import cv2
import dlib
import numpy as np
import scipy
from scipy.spatial import distance
import time
import pygame
import threading  
import logging
logging.basicConfig(level=logging.WARNING)

kivy.require('2.0.0')
np.require('1.26.4')



def drowsiness_detection():
    pygame.init()
    pygame.mixer.init()
    warning_sound = pygame.mixer.Sound("music.wav")  
    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()  
    predictor = dlib.shape_predictor(shape_predictor_path) 

    def get_eye_aspect_ratio(eye):
        
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    EAR_THRESHOLD = 0.25  
    EAR_CONSEC_FRAMES = 20  
    COUNTER = 0  

    while True:
        ret, frame = cap.read()  

        if not ret:  
            print("Error: Failed to capture frame.")
            break

        
        if frame is None or frame.dtype != np.uint8:
            print("Error: Frame is not in the expected 8-bit format.") 
            continue

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     
        if gray.dtype != np.uint8:
            print("Error: Grayscale image is not 8-bit.")  
            continue

        
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)  
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            left_eye = landmarks[36:42]  
            right_eye = landmarks[42:48] 
            
            left_ear = get_eye_aspect_ratio(left_eye)
            right_ear = get_eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) // 2.0

           
            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES:
                    warning_sound.play() 
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0

            
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        cv2.imshow("Frame", frame)  

        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    cap.release()  
    cv2.destroyAllWindows()  



class MyButton(FloatLayout):

    def __init__(self, **kwargs):
        super(MyButton, self).__init__(**kwargs)

        with self.canvas.before:
            Color(247, 193, 15, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self._update_rect, pos=self._update_rect)

        start_button = Button(
            text="START",
            font_size=50,
            size_hint=(0.7, 0.15),
            size=(50, 100),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            background_color=(1, 0, 0,0.7) 
        )
        start_button.bind(on_press=self.press)
        self.add_widget(start_button)

    def press(self, instance):
        print("Starting drowsiness detection...")
       
        threading.Thread(target=drowsiness_detection, daemon=True).start()

    def _update_rect(self, instance, value):
        self.rect.pos = self.pos
        self.rect.size = self.size


class MyApp(App):
    def build(self):
        return MyButton()


if __name__ == '__main__':
    MyApp().run()
