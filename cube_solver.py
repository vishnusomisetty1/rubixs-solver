# cube_solver.py
import logging
import kociemba
import serial
import time
from color_detection import ColorDetector, manual_input

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
            # Calibrate colors
            logging.info("Starting color calibration...")
            if not self.color_detector.calibrate_all_colors():
                logging.error("Calibration failed.")
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
                    logging.info(f"\nPrepare to show the {side} side of the cube.")
                    colors = self.color_detector.scan_cube_face()
                    if colors is None:
                        logging.error("Failed to scan cube face.")
                        return
                    
                    confirm = input("Are the colors correct? (Y/N): ").strip().upper()
                    if confirm != 'Y':
                        logging.info("Rescan the side.")
                        side -= 1  # Redo this side
                        continue
                    
                    cube_state += colors
            
            # Solve and execute
            logging.info("Detected cube state: %s", ''.join(cube_state))
            solution = self.solve_cube(cube_state)
            
            if solution:
                logging.info(f"Solution: {solution}")
                self.send_to_arduino(solution)
            else:
                logging.error("Failed to find a solution.")
        
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