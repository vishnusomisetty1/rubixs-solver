import cv2
import time
import logging
from config import *

class CameraManager:
    def __init__(self):
        self.camera = None
    
    def initialize(self):
        logging.info("Initializing camera...")
        for i in range(3):  # Try 3 times
            try:
                self.camera = cv2.VideoCapture(0)
                if self.camera.isOpened():
                    logging.info("Camera initialized successfully.")
                    return True
                else:
                    logging.warning(f"Attempt {i+1}: Failed to open camera.")
            except Exception as e:
                logging.error(f"Attempt {i+1}: Error initializing camera: {e}")
            time.sleep(1)  # Wait for 1 second before retrying
        logging.error("Failed to initialize camera after 3 attempts.")
        return False
    
    def read_frame(self):
        if self.camera:
            return self.camera.read()
        return False, None
    
    def release(self):
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()