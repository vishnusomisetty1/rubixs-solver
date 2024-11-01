import cv2
import numpy as np
import logging
from ui_helpers import UIHelper 

class ColorDetector:
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        self.calibrated_colors = {}
    
    def calibrate_color(self, color_name):
        logging.info(f"Calibrating {color_name}. Show the {color_name} side of the cube.")
        
        button_pos = (10, 10)
        button_size = (200, 50)
        clicked = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                if UIHelper.is_button_clicked((x, y), button_pos, button_size):
                    clicked = True
        
        cv2.namedWindow('Color Calibration')
        cv2.setMouseCallback('Color Calibration', mouse_callback)
        
        while not clicked:
            ret, frame = self.camera_manager.read_frame()
            if not ret:
                logging.error("Failed to capture frame.")
                return None
            
            UIHelper.create_button(frame, "Capture", button_pos, button_size)
            cv2.imshow('Color Calibration', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        height, width = frame.shape[:2]
        center_roi = frame[height//3:2*height//3, width//3:2*width//3]
        average_color = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV).mean(axis=(0,1))
        
        logging.info(f"{color_name} calibrated successfully.")
        return average_color

    def calibrate_all_colors(self):
        colors = ['White', 'Red', 'Green', 'Blue', 'Orange', 'Yellow']
        for color in colors:
            color_value = self.calibrate_color(color)
            if color_value is None:
                logging.error("Calibration cancelled.")
                return False
            self.calibrated_colors[color] = color_value
        return True