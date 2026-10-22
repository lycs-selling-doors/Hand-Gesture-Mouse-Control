"""Webcam gesture mouse control using MediaPipe Tasks and OpenCV."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui

if __package__:
    from .gesture_logic import MovementGate, is_move_gesture
    from .hand_tracking import HandLandmark, HandTracker, draw_landmarks
    from .model import DEFAULT_MODEL_PATH
else:
    from gesture_logic import MovementGate, is_move_gesture
    from hand_tracking import HandLandmark, HandTracker, draw_landmarks
    from model import DEFAULT_MODEL_PATH

# Keep the original gesture thresholds and mouse smoothing behavior.
gesture_distance_threshold = 20
click_delay = 0.3
last_gesture_time = time.time()
smoothing_factor = 0.5

class MouseControl:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.last_smooth_x = screen_width / 2
        self.last_smooth_y = screen_height / 2
        self.last_move_time = time.time()

    def move_mouse(self, x, y, frame_width, frame_height):
        move_interval = 0.05

        if time.time() - self.last_move_time >= move_interval:
            # 增加垂直缩放系数
            vertical_scale = 2.0  # 从1.5增加到2.0
            horizontal_scale = 1.2

            # 修改Y轴映射方式，添加偏移量使手势范围向上延伸
            y_offset = frame_height * 0.2  # 添加20%的向上偏移
            adjusted_y = max(0, y - y_offset)
            
            target_mouse_x = np.clip((self.screen_width - (x / frame_width) * self.screen_width * horizontal_scale), 0, self.screen_width)
            target_mouse_y = np.clip((adjusted_y / frame_height) * self.screen_height * vertical_scale, 0, self.screen_height)

            # 使用更激进的非线性映射
            target_mouse_y = self._apply_nonlinear_mapping(target_mouse_y, self.screen_height)

            # 使用平滑（EMA）
            smooth_x = self.last_smooth_x * (1 - smoothing_factor) + target_mouse_x * smoothing_factor
            smooth_y = self.last_smooth_y * (1 - smoothing_factor) + target_mouse_y * smoothing_factor

            # 确保坐标在屏幕范围内
            smooth_x = np.clip(smooth_x, 0, self.screen_width)
            smooth_y = np.clip(smooth_y, 0, self.screen_height)

            self.last_smooth_x = smooth_x
            self.last_smooth_y = smooth_y
            pyautogui.moveTo(smooth_x, smooth_y)
            self.last_move_time = time.time()

    def _apply_nonlinear_mapping(self, y, max_height):
        """使用更激进的非线性映射"""
        normalized_y = y / max_height
        # 使用更小的指数来增强上部区域的精确度
        mapped_y = normalized_y ** 1.2  # 从1.5减小到1.2
        return mapped_y * max_height
        
        
# 设置点击动作
def left_click():
    pyautogui.click(button='left')

def right_click():
    pyautogui.click(button='right')

def press_key(key):
    """模拟按下键盘按键"""
    pyautogui.press(key)

def double_click():
    """模拟双击鼠标左键"""
    pyautogui.doubleClick(button='left')

def scroll_up():
    """模拟鼠标滚轮向上滚动"""
    pyautogui.scroll(10)  # 滚动向上

def scroll_down():
    """模拟鼠标滚轮向下滚动"""
    pyautogui.scroll(-10)  # 滚动向下


def start_hand_gesture_control(model_path=DEFAULT_MODEL_PATH):
    global last_gesture_time
    movement = MovementGate()

    # Validate/load the model before taking ownership of the camera.
    with HandTracker(model_path) as hands:
        cap = cv2.VideoCapture(0)
        try:
            if not cap.isOpened():
                raise RuntimeError("Could not open camera 0.")
            mouse_control = MouseControl(*pyautogui.size())
            cv2.namedWindow("Hand Gesture Control")
            while True:
                # 读取每一帧
                ret, frame = cap.read()

                if not ret:
                    print("Failed to capture image from camera.")
                    break

                # 获取视频帧的宽度和高度
                frame_height, frame_width, _ = frame.shape

                # 使用 OpenCV 转换为 RGB 图像
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 使用 MediaPipe 进行手部检测
                results = hands.process(rgb_frame)

                # 如果检测到手部
                if results.hand_landmarks:
                    for hand_landmarks in results.hand_landmarks:
                        # 获取手的关键点
                        landmarks = hand_landmarks
                        # 绘制手部关键点
                        draw_landmarks(frame, hand_landmarks)

                        # 获取五个手指尖的坐标
                        thumb_tip = landmarks[HandLandmark.THUMB_TIP]
                        index_finger_tip = landmarks[HandLandmark.INDEX_FINGER_TIP]
                        middle_finger_tip = landmarks[HandLandmark.MIDDLE_FINGER_TIP]
                        ring_finger_tip = landmarks[HandLandmark.RING_FINGER_TIP]
                        pinky_tip = landmarks[HandLandmark.PINKY_TIP]

                        # 获取屏幕上的坐标
                        thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
                        index_x, index_y = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)
                        middle_x, middle_y = int(middle_finger_tip.x * frame_width), int(middle_finger_tip.y * frame_height)
                        ring_x, ring_y = int(ring_finger_tip.x * frame_width), int(ring_finger_tip.y * frame_height)
                        pinky_x, pinky_y = int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)

                        # Only move while both fingers are extended and together.
                        held = is_move_gesture(landmarks, frame_width / frame_height)
                        if movement.update(held, time.monotonic()):
                            mouse_control.move_mouse(index_x, index_y, frame_width, frame_height)

                        # 判断拇指与食指是否对碰，模拟左键单击
                        thumb_index_distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
                        if thumb_index_distance < gesture_distance_threshold:
                            if time.time() - last_gesture_time > click_delay:
                                left_click()
                                last_gesture_time = time.time()

                        # 判断大拇指与中指是否对碰，模拟滑轮向下滚动
                        thumb_middle_distance = np.sqrt((thumb_x - middle_x) ** 2 + (thumb_y - middle_y) ** 2)
                        if thumb_middle_distance < gesture_distance_threshold:
                            # 手指保持叠加，持续向下滚动
                            scroll_down()  # 模拟滑轮向下滚动
                            time.sleep(0.0)  # 滚动的间隔时间
                            last_gesture_time = time.time()
                        else:
                            # 如果手指分开，停止滚动
                            pass

                        # 判断大拇指与无名指是否对碰，模拟滑轮向上滚动
                        thumb_ring_distance = np.sqrt((thumb_x - ring_x) ** 2 + (thumb_y - ring_y) ** 2)
                        if thumb_ring_distance < gesture_distance_threshold:
                            # 手指保持叠加，持续向上滚动
                            scroll_up()  # 模拟滑轮向上滚动
                            time.sleep(0)  # 滚动的间隔时间
                            last_gesture_time = time.time()
                        else:
                            # 如果手指分开，停止滚动
                            pass

                        # 判断大拇指与小指是否对碰，模拟双击鼠标左键
                        thumb_pinky_distance = np.sqrt((thumb_x - pinky_x) ** 2 + 2 * (thumb_y - pinky_y) ** 2)
                        if thumb_pinky_distance < gesture_distance_threshold:
                            if time.time() - last_gesture_time > click_delay:
                                double_click()  # 模拟双击
                                last_gesture_time = time.time()

                if not results.hand_landmarks:
                    movement.reset()

                # 显示带有手部标记的图像
                cv2.imshow("Hand Gesture Control", frame)

                # 按 'q' 键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Path to a local hand_landmarker.task model")
    args = parser.parse_args()
    start_hand_gesture_control(args.model)
