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

    def optimize_moves(self, solution):
        """Optimize the move sequence by combining consecutive moves on the same face"""
        moves = solution.split()
        optimized = []
        i = 0
        
        while i < len(moves):
            if i + 1 < len(moves) and moves[i][0] == moves[i + 1][0]:
                # Same face moves
                face = moves[i][0]
                
                # Count moves
                count = 0
                if len(moves[i]) == 1:  # Single turn
                    count += 1
                elif moves[i][1] == '2':  # Double turn
                    count += 2
                elif moves[i][1] == "'":  # Counter-clockwise
                    count += 3
                
                if len(moves[i + 1]) == 1:
                    count += 1
                elif moves[i + 1][1] == '2':
                    count += 2
                elif moves[i + 1][1] == "'":
                    count += 3
                
                # Normalize count to 0-3
                count = count % 4
                
                # Convert count to move
                if count == 0:
                    # Moves cancel out
                    pass
                elif count == 1:
                    optimized.append(face)
                elif count == 2:
                    optimized.append(face + '2')
                else:  # count == 3
                    optimized.append(face + "'")
                
                i += 2
            else:
                optimized.append(moves[i])
                i += 1
        
        return ' '.join(optimized)

    def send_to_arduino(self, solution):
        """Send solution to Arduino with improved timing and verification"""
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
                
                logging.info("Arduino is ready. Sending solution...")
                
                # Optimize the solution
                optimized_solution = self.optimize_moves(solution)
                moves = optimized_solution.split()
                
                logging.info(f"Original solution: {solution}")
                logging.info(f"Optimized solution: {optimized_solution}")
                
                # Send moves one at a time with improved timing
                for idx, move in enumerate(moves):
                    logging.info(f"Sending move: {move}")
                    command = f"{move}\n"
                    ser.write(command.encode())
                    
                    # Wait for move completion with timeout
                    start_time = time.time()
                    move_completed = False
                    
                    while time.time() - start_time < 5:  # 5 second timeout
                        if ser.in_waiting:
                            response = ser.readline().decode().strip()
                            logging.info(f"Arduino: {response}")
                            
                            if response == "MOVE_COMPLETE":
                                move_completed = True
                                break
                            elif response.startswith("ERROR"):
                                logging.error(f"Error occurred: {response}")
                                return False
                            elif response.startswith("EXECUTING"):
                                logging.info(f"Executing move: {move}")
                    
                    if not move_completed:
                        logging.error(f"Move {move} timed out!")
                        return False
                    
                    # Base delay
                    time.sleep(0.75)  # 0.75 second base delay
                    
                    # Additional delays based on move type
                    if '2' in move:  # Double moves
                        time.sleep(0.25)
                    elif "'" in move:  # Counter-clockwise moves
                        time.sleep(0.15)
                    
                    # Extra delay after potentially problematic sequences
                    if idx > 0 and move[0] == moves[idx-1][0]:  # Same face moves
                        time.sleep(0.2)
                
                logging.info("Solution sent to Arduino successfully.")
                return True
            
        except serial.SerialException as e:
            logging.error(f"Error communicating with Arduino: {e}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
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
            
            current_face = 0
            while current_face < len(faces):
                face_name, center_color = faces[current_face]
                logging.info(f"\nScanning {face_name}...")
                print(f"\nPosition the cube with the {face_name} center facing the camera")
                
                colors = self.color_detector.scan_cube_face(face_name, center_color)
                if colors is None:  # User pressed ESC
                    logging.info("Scanning cancelled.")
                    return
                
                print("\nDetected colors:")
                self.color_detector.display_colors(colors)
                
                while True:
                    choice = input("\nChoose action:\n1. Proceed to next side\n2. Manual input\n3. Quit\nChoice (1/2/3): ").strip()
                    if choice in ['1', '2', '3']:
                        break
                    print("Invalid choice. Please try again.")
                
                if choice == '1':  # Proceed to next side
                    cube_state.extend(colors)
                    current_face += 1
                elif choice == '2':  # Manual input
                    print(f"\nEnter colors for {face_name} (use R,O,Y,G,B,W):")
                    colors = self.color_detector.manual_input()
                    colors[4] = center_color  # Ensure center color is correct
                    cube_state.extend(colors)
                    current_face += 1
                else:  # Quit
                    logging.info("Program terminated by user.")
                    return
            
            if len(cube_state) == 54:  # Complete cube
                # Solve and get solution
                logging.info("Detected cube state: %s", ''.join(cube_state))
                solution = self.solve_cube(cube_state)
                
                if solution:
                    logging.info(f"\nSolution found: {solution}")
                    
                    while True:
                        choice = input("\nChoose action:\n1. Send solution to Arduino\n2. Exit\nChoice (1/2): ").strip()
                        if choice in ['1', '2']:
                            break
                        print("Invalid choice. Please try again.")
                    
                    if choice == '1':
                        logging.info("Sending solution to Arduino...")
                        if self.send_to_arduino(solution):
                            logging.info("Solution successfully sent to Arduino.")
                        else:
                            logging.error("Failed to send solution to Arduino.")
                    else:
                        logging.info("Exiting without sending to Arduino.")
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