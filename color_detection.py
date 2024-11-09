# color_detection.py
import cv2
import numpy as np
import logging
import time

class ColorDetector:
    def __init__(self):
        self.camera = None
        # Reset HSV ranges with wider tolerances
        self.color_ranges = {
            'W': [(np.array([0, 0, 200]), np.array([180, 30, 255]))],    # White - high value, low saturation
            'R': [(np.array([0, 150, 150]), np.array([10, 255, 255])),   # Red lower range
                  (np.array([170, 150, 150]), np.array([180, 255, 255]))], # Red upper range
            'O': [(np.array([10, 150, 150]), np.array([20, 255, 255]))], # Orange
            'Y': [(np.array([20, 100, 150]), np.array([30, 255, 255]))], # Yellow - lower saturation requirement
            'G': [(np.array([45, 150, 150]), np.array([75, 255, 255]))], # Green - moved away from yellow
            'B': [(np.array([95, 150, 150]), np.array([115, 255, 255]))] # Blue - narrower range
        }
        
        # Define BGR colors for visualization
        self.color_bgr = {
            'W': (255, 255, 255),  # White
            'R': (0, 0, 255),      # Red
            'O': (0, 128, 255),    # Orange
            'Y': (0, 255, 255),    # Yellow
            'G': (0, 255, 0),      # Green
            'B': (255, 0, 0)       # Blue
        }

    def initialize_camera(self):
        """Initialize the camera with multiple attempts and set resolution"""
        logging.info("Initializing camera...")
        for i in range(3):
            try:
                self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Added CAP_DSHOW for Windows
                if self.camera.isOpened():
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    
                    actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    logging.info(f"Camera initialized at: {actual_width}x{actual_height}")
                    return True
                logging.warning(f"Attempt {i+1}: Failed to open camera.")
            except Exception as e:
                logging.error(f"Attempt {i+1}: Error initializing camera: {e}")
            time.sleep(1)
        logging.error("Failed to initialize camera after 3 attempts.")
        return False

    def release_camera(self):
        """Release the camera and close all windows"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

    def draw_cube_guide(self, frame):
        """Draw a 3x3 grid guide on the frame"""
        height, width = frame.shape[:2]
        square_size = min(height, width) // 4
        start_x = (width - 3*square_size) // 2
        start_y = (height - 3*square_size) // 2
        
        # Draw the grid
        for i in range(4):
            x = start_x + i * square_size
            cv2.line(frame, (x, start_y), (x, start_y + 3*square_size), (255, 255, 255), 2)
            y = start_y + i * square_size
            cv2.line(frame, (start_x, y), (start_x + 3*square_size, y), (255, 255, 255), 2)
        
        return frame, start_x, start_y, square_size

    def get_color_name(self, hsv_value):
        """Determine color name with simplified logic and strict thresholds"""
        h, s, v = hsv_value

        # White detection first (high value, low saturation)
        if v > 200 and s < 30:
            return 'W'

        # For all other colors, require minimum saturation and value
        if s < 100 or v < 150:
            return 'W'  # If saturation or value is too low, consider it white

        # Red detection (handle wraparound)
        if (h <= 10) or (h >= 170):
            if s > 150:
                return 'R'

        # Other colors based on hue ranges
        if 10 <= h <= 20:
            return 'O'
        elif 20 <= h <= 30:
            return 'Y'
        elif 45 <= h <= 75:
            return 'G'
        elif 95 <= h <= 115:
            return 'B'

        # Find closest match if no direct match found
        min_distance = float('inf')
        best_match = None
        
        # Define hue centers for each color for distance calculation
        hue_centers = {
            'R': 5,    # Also check 175
            'O': 15,
            'Y': 25,
            'G': 60,
            'B': 105
        }

        for color, center in hue_centers.items():
            if color == 'R':
                # Special handling for red's wraparound
                dist = min(abs(h - center), abs(h - 175))
            else:
                dist = abs(h - center)
            
            if dist < min_distance:
                min_distance = dist
                best_match = color

        return best_match

    def detect_cube_colors(self, frame, center_color):
        """Detect colors with improved sampling and noise reduction"""
        height, width = frame.shape[:2]
        square_size = min(height, width) // 4
        start_x = (width - 3*square_size) // 2
        start_y = (height - 3*square_size) // 2
        
        colors = []
        
        # Convert to HSV and apply preprocessing
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_frame = cv2.GaussianBlur(hsv_frame, (5, 5), 0)
        
        # Enhance contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        hsv_frame = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        for i in range(3):
            for j in range(3):
                is_center = (i == 1 and j == 1)
                
                if is_center:
                    colors.append(center_color)
                else:
                    center_x = start_x + j * square_size + square_size // 2
                    center_y = start_y + i * square_size + square_size // 2
                    
                    # Sample a larger area
                    sample_size = 5
                    color_votes = {}
                    
                    for dy in range(-sample_size, sample_size + 1):
                        for dx in range(-sample_size, sample_size + 1):
                            sample_x = center_x + dx * 2
                            sample_y = center_y + dy * 2
                            
                            if 0 <= sample_x < width and 0 <= sample_y < height:
                                hsv_value = hsv_frame[sample_y, sample_x]
                                color = self.get_color_name(hsv_value)
                                color_votes[color] = color_votes.get(color, 0) + 1
                    
                    if color_votes:
                        # Require a more decisive majority
                        total_votes = sum(color_votes.values())
                        best_color = max(color_votes.items(), key=lambda x: x[1])
                        if best_color[1] / total_votes >= 0.6:  # Increased threshold to 60%
                            color = best_color[0]
                        else:
                            # If no clear majority, use center pixel
                            color = self.get_color_name(hsv_frame[center_y, center_x])
                    else:
                        color = self.get_color_name(hsv_frame[center_y, center_x])
                    
                    colors.append(color)
                
                # Draw visualization
                rect_x = start_x + j * square_size
                rect_y = start_y + i * square_size
                
                indicator_size = square_size // 3
                indicator_x = rect_x + (square_size - indicator_size) // 2
                indicator_y = rect_y + (square_size - indicator_size) // 2
                
                border_color = (0, 255, 0) if is_center else (255, 255, 255)
                cv2.rectangle(frame, (rect_x, rect_y), 
                            (rect_x + square_size, rect_y + square_size),
                            border_color, 2)
                
                color_bgr = self.color_bgr[colors[-1]]
                cv2.rectangle(frame, 
                            (indicator_x, indicator_y),
                            (indicator_x + indicator_size, indicator_y + indicator_size),
                            color_bgr, -1)
                
                cv2.rectangle(frame, 
                            (indicator_x, indicator_y),
                            (indicator_x + indicator_size, indicator_y + indicator_size),
                            (0, 0, 0), 1)
        
        return colors, frame

    def scan_cube_face(self, face_name, center_color):
        """Scan a single face of the cube"""
        while True:
            ret, frame = self.camera.read()
            if not ret:
                logging.error("Failed to capture frame.")
                return None
            
            # Draw the guide grid and get dimensions
            frame_with_guide, start_x, start_y, square_size = self.draw_cube_guide(frame.copy())
            
            # Real-time color detection with known center color
            colors, frame_with_colors = self.detect_cube_colors(frame.copy(), center_color)
            
            # Add instructions
            cv2.putText(frame_with_colors, f"Scanning {face_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_with_colors, "Press SPACE to capture, ESC to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('Cube Face Detection', frame_with_colors)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                return None
            elif key == ord(' '):  # Space key
                cv2.destroyAllWindows()
                return colors

    @staticmethod
    def display_colors(colors):
        """Display the detected colors in a grid format"""
        for i in range(0, 9, 3):
            print(f"{colors[i]} {colors[i+1]} {colors[i+2]}")

    @staticmethod
    def manual_input():
        """Get manual color input for a cube face"""
        while True:
            colors = input("Enter 9 colors (use R,O,Y,G,B,W): ").upper().strip()
            if len(colors) == 9 and all(c in 'ROYGBW' for c in colors):
                return list(colors)
            print("Invalid input. Please enter exactly 9 colors using R, O, Y, G, B, W.")

def test_color_detection():
    logging.basicConfig(level=logging.INFO)
    detector = ColorDetector()
    
    # Define the faces by their center colors
    faces = [
        ("White face", "W"),
        ("Red face", "R"),
        ("Green face", "G"),
        ("Yellow face", "Y"),
        ("Orange face", "O"),
        ("Blue face", "B")
    ]
    
    if detector.initialize_camera():
        try:
            all_colors = []
            
            for face_name, center_color in faces:
                logging.info(f"\nScanning {face_name}...")
                print(f"\nPosition the cube with the {face_name} center facing the camera")
                colors = detector.scan_cube_face(face_name, center_color)
                if colors:
                    detector.display_colors(colors)
                    all_colors.extend(colors)
                else:
                    logging.error(f"Failed to scan {face_name}")
                    break
            
            if len(all_colors) == 54:  # Complete cube
                logging.info("Complete cube state: %s", ''.join(all_colors))
            
        finally:
            detector.release_camera()

if __name__ == "__main__":
    test_color_detection()

# cube_solver.py
import logging
import kociemba
import serial
import time
from color_detection import ColorDetector

class CubeSolver:
    def __init__(self, arduino_port='COM3', baud_rate=9600):
        self.arduino_port = arduino_port
        self.baud_rate = baud_rate
        self.color_detector = ColorDetector()

    def initialize(self):
        """Initialize the system"""
        return self.color_detector.initialize_camera()

    def cleanup(self):
        """Clean up resources"""
        self.color_detector.release_camera()

    def solve_cube(self, cube_state):
        """Solve the cube using Kociemba algorithm"""
        try:
            kociemba_state = self._convert_to_kociemba_format(cube_state)
            solution = kociemba.solve(kociemba_state)
            return solution
        except Exception as e:
            logging.error(f"Error solving cube: {e}")
            return None

    @staticmethod
    def _convert_to_kociemba_format(cube_state):
        """Convert our color format to Kociemba format"""
        color_mapping = {
            'W': 'U', 'R': 'R', 'G': 'F',
            'Y': 'D', 'O': 'L', 'B': 'B'
        }
        return ''.join(color_mapping[c] for c in cube_state)

    def send_to_arduino(self, solution):
        """Send solution to Arduino"""
        try:
            with serial.Serial(self.arduino_port, self.baud_rate, timeout=1) as ser:
                logging.info("Connecting to Arduino...")
                time.sleep(2)  # Wait for Arduino to reset
                
                # Wait for Arduino to be ready
                while True:
                    if ser.in_waiting:
                        response = ser.readline().decode().strip()
                        logging.info(f"Arduino: {response}")
                        if response == "READY":
                            break
                logging.info("Arduino is ready. Sending calibration command...")
                ser.write(b"CALIBRATE\n")
                
                # Wait for calibration to complete
                while True:
                    if ser.in_waiting:
                        response = ser.readline().decode().strip()
                        logging.info(f"Arduino: {response}")
                        if response == "CALIBRATION_COMPLETE":
                            break
                
                logging.info("Calibration complete. Sending solution...")
                moves = solution.split()
                for move in moves:
                    logging.info(f"Sending move: {move}")
                    ser.write(f"{move}\n".encode())
                    
                    # Wait for move to complete
                    while True:
                        if ser.in_waiting:
                            response = ser.readline().decode().strip()
                            logging.info(f"Arduino: {response}")
                            if response.startswith("MOVE_COMPLETE"):
                                break
                            elif response.startswith("ERROR"):
                                logging.error(f"Error occurred: {response}")
                                return False
                    
                    time.sleep(0.1)  # Small delay between moves
            
            logging.info("Solution sent to Arduino successfully.")
            return True
        except serial.SerialException as e:
            logging.error(f"Error communicating with Arduino: {e}")
            return False

    def run(self):
        """Main execution flow"""
        try:
            # Process each side of the cube
            cube_state = []
            faces = [
                ("White face", "W"),
                ("Red face", "R"),
                ("Green face", "G"),
                ("Yellow face", "Y"),
                ("Orange face", "O"),
                ("Blue face", "B")
            ]
            
            for face_name, center_color in faces:
                while True:
                    logging.info(f"\nScanning {face_name}...")
                    print(f"\nPosition the cube with the {face_name} center facing the camera")
                    colors = self.color_detector.scan_cube_face(face_name, center_color)
                    
                    if colors is None:  # User pressed ESC
                        logging.info("Scanning cancelled.")
                        return
                    
                    print("\nDetected colors:")
                    self.color_detector.display_colors(colors)
                    
                    while True:
                        choice = input("\nChoose action:\n1. Proceed with detected colors\n2. Retry scan\n3. Manual input\n4. Quit\nChoice (1/2/3/4): ").strip()
                        if choice in ['1', '2', '3', '4']:
                            break
                        print("Invalid choice. Please try again.")
                    
                    if choice == '1':  # Proceed
                        cube_state.extend(colors)
                        break
                    elif choice == '2':  # Retry
                        continue
                    elif choice == '3':  # Manual input
                        print(f"\nEnter colors for {face_name} (use R,O,Y,G,B,W):")
                        colors = self.color_detector.manual_input()
                        colors[4] = center_color  # Ensure center color is correct
                        cube_state.extend(colors)
                        break
                    else:  # Quit
                        logging.info("Program terminated by user.")
                        return
                    
                if choice == '4':  # If user chose to quit
                    return
            
            if len(cube_state) == 54:  # Complete cube
                # Solve and execute
                logging.info("Detected cube state: %s", ''.join(cube_state))
                solution = self.solve_cube(cube_state)
                
                if solution:
                    logging.info(f"Solution: {solution}")
                    self.send_to_arduino(solution)
                else:
                    logging.error("Failed to find a solution.")
            else:
                logging.error("Incomplete cube state detected.")
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create and run solver
    solver = CubeSolver()
    if solver.initialize():
        solver.run()
    solver.cleanup()

if __name__ == "__main__":
    main()