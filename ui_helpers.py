import cv2

class UIHelper:
    @staticmethod
    def create_button(img, text, position, size):
        cv2.rectangle(img, position, (position[0] + size[0], position[1] + size[1]), (0, 255, 0), -1)
        cv2.putText(img, text, (position[0] + 10, position[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    @staticmethod
    def is_button_clicked(mouse_pos, button_pos, button_size):
        return (button_pos[0] < mouse_pos[0] < button_pos[0] + button_size[0] and
                button_pos[1] < mouse_pos[1] < button_pos[1] + button_size[1])