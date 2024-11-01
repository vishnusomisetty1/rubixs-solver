import serial
import time
import logging
from config import *

class ArduinoController:
    def __init__(self, port=ARDUINO_PORT, baud_rate=BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        
    def send_solution(self, solution):
        try:
            with serial.Serial(self.port, self.baud_rate, timeout=1) as ser:
                logging.info("Connecting to Arduino...")
                time.sleep(2)  # Wait for Arduino to reset
                
                # Wait for Arduino to be ready
                self._wait_for_response(ser, "READY")
                
                logging.info("Arduino is ready. Sending calibration command...")
                ser.write(b"CALIBRATE\n")
                
                # Wait for calibration to complete
                self._wait_for_response(ser, "CALIBRATION_COMPLETE")
                
                logging.info("Calibration complete. Sending solution...")
                moves = solution.split()
                for move in moves:
                    logging.info(f"Sending move: {move}")
                    ser.write(f"{move}\n".encode())
                    self._wait_for_move_completion(ser)
                    time.sleep(0.1)  # Small delay between moves
            
            logging.info("Solution sent to Arduino successfully.")
        except serial.SerialException as e:
            logging.error(f"Error communicating with Arduino: {e}")
            
    def _wait_for_response(self, ser, expected_response):
        while True:
            if ser.in_waiting:
                response = ser.readline().decode().strip()
                logging.info(f"Arduino: {response}")
                if response == expected_response:
                    break
                    
    def _wait_for_move_completion(self, ser):
        while True:
            if ser.in_waiting:
                response = ser.readline().decode().strip()
                logging.info(f"Arduino: {response}")
                if response.startswith("MOVE_COMPLETE"):
                    break
                elif response.startswith("ERROR"):
                    logging.error(f"Error occurred: {response}")
                    raise Exception("Arduino reported an error")