import kociemba
import logging

class CubeSolver:
    @staticmethod
    def convert_to_kociemba_format(cube_state):
        color_mapping = {
            'W': 'U', 'R': 'R', 'G': 'F',
            'Y': 'D', 'O': 'L', 'B': 'B'
        }
        return ''.join(color_mapping[c] for c in cube_state)

    @staticmethod
    def solve(cube_state):
        try:
            kociemba_state = CubeSolver.convert_to_kociemba_format(cube_state)
            solution = kociemba.solve(kociemba_state)
            return solution
        except Exception as e:
            logging.error(f"Error solving cube: {e}")
            return None