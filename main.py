from camera import CameraManager
from color_detector import ColorDetector
from solver import CubeSolver
from arduino_controller import ArduinoController
from ui_helpers import UIHelper
import cv2
import logging

def manual_input():
    while True:
        colors = input("Enter 9 colors (use R,O,Y,G,B,W): ").upper().strip()
        if len(colors) == 9 and all(c in 'ROYGBW' for c in colors):
            return list(colors)
        logging.error("Invalid input. Please enter exactly 9 colors using R, O, Y, G, B, W.")

def main():
    # Initialize components
    camera_manager = CameraManager()
    if not camera_manager.initialize():
        logging.error("Exiting program due to camera initialization failure.")
        return

    try:
        color_detector = ColorDetector(camera_manager)
        solver = CubeSolver()
        arduino_controller = ArduinoController()

        # Calibrate colors
        logging.info("Starting color calibration...")
        if not color_detector.calibrate_all_colors():
            logging.error("Calibration failed. Exiting program.")
            return
        logging.info("Calibration complete.")
        
        # Process each side of the cube
        cube_state = []
        sides = ['Up', 'Right', 'Front', 'Down', 'Left', 'Back']
        
        use_manual_input = input("Do you want to manually input the cube colors? (Y/N): ").strip().upper()

        for side in sides:
            if use_manual_input == 'Y':
                logging.info(f"\nManually inputting the {side} side of the cube.")
                cube_state += manual_input()
            else:
                # Implement camera-based detection here
                pass  # Add your camera-based detection logic

        # Solve and execute
        logging.info("Detected cube state: %s", ''.join(cube_state))
        solution = solver.solve(cube_state)
        
        if solution:
            logging.info(f"Solution: {solution}")
            arduino_controller.send_solution(solution)
        else:
            logging.error("Failed to find a solution.")
    
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    
    finally:
        camera_manager.release()

if __name__ == "__main__":
    main()