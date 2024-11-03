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
            cv2.putText(frame_with_colors, f"Align face with {face_name} in the middle", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_with_colors, "Press SPACE to capture or ESC to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('Cube Face Detection', frame_with_colors)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cv2.destroyAllWindows()
                return None
            elif key == 32:  # SPACE key
                cv2.destroyAllWindows()
                
                # Ask for confirmation
                print("\nDetected colors:")
                self.display_colors(colors)
                while True:
                    choice = input("\nChoose action:\n1. Accept and continue\n2. Retry scan\n3. Manual input\nChoice (1/2/3): ").strip()
                    if choice in ['1', '2', '3']:
                        break
                
                if choice == '1':
                    return colors
                elif choice == '2':
                    continue
                else:  # choice == '3'
                    colors = self.manual_input()
                    # Ensure center color is correct even in manual input
                    colors[4] = center_color  # Position 4 is the center (0-based indexing)
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