"""Camera worker and a narrow JS bridge for the local pywebview frontend."""

import base64
from copy import deepcopy
import ctypes
import platform
import threading
import time
import uuid

import cv2
import numpy as np

from .control_config import ConfigStore, GESTURES, POSTURES, PROFILES, validate_config
from .download_model import download_model
from .gesture_logic import GestureGate, PointerMapper, SensitivityTuner, pose_features, recognize
from .hand_tracking import HandTracker, draw_landmarks
from .model import DEFAULT_MODEL_PATH


def cuda_capability():
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count <= 0 or not hasattr(cv2.cuda, "cvtColor"):
            return {"available": False, "devices": max(count, 0),
                    "reason": "当前 OpenCV 没有可用的 CUDA 图像处理后端。需 NVIDIA 驱动及带 CUDA 的 OpenCV 构建。"}
        # A device count alone does not prove the required CUDA module works.
        mat = cv2.cuda_GpuMat()
        mat.upload(np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.cuda.cvtColor(mat, cv2.COLOR_BGR2RGB).download()
        return {"available": True, "devices": count, "reason": "可启用 CUDA 颜色转换；手部模型仍使用 CPU。"}
    except Exception as exc:
        return {"available": False, "devices": 0, "reason": f"CUDA 检测失败：{exc}"}


def dispatch(action, automation):
    if action == "none":
        return
    kind, _, value = action.partition(":")
    if kind == "click":
        automation.click(button=value, _pause=False)
    elif kind == "doubleclick":
        automation.doubleClick(interval=.08, _pause=False)
    elif kind == "scroll":
        automation.scroll(int(value), _pause=False)
    elif kind == "key":
        automation.press(value, _pause=False)
    elif kind == "hotkey":
        pressed = []
        try:
            for key in value.split("+"):
                pressed.append(key)
                automation.keyDown(key, _pause=False)
        finally:
            fail_safe = automation.FAILSAFE
            try:
                # Release only: the corner fail-safe must not strand Ctrl/Alt.
                automation.FAILSAFE = False
                for key in reversed(pressed):
                    automation.keyUp(key, _pause=False)
            finally:
                automation.FAILSAFE = fail_safe


class DesktopController:
    def __init__(self, store=None, automation=None, model_path=DEFAULT_MODEL_PATH):
        self._store = store or ConfigStore()
        self._config = self._store.load()
        self._automation = automation
        self._model_path = model_path
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._cuda = cuda_capability()
        self._capture = None
        self._arm_at = 0
        self._last_hand = None
        self._last_hand_at = 0
        self._revision = 0
        self._gui_focused = True
        self._state = {"running": False, "armed": False, "arming": 0, "tuning": "",
                       "frame": "", "frame_id": 0, "fps": 0, "landmarks": [],
                       "gesture": "等待手势", "last_action": "尚未输出操作", "error": self._store.error,
                       "capture": "", "downloading": False, "preprocessing": "CPU", "inference": "CPU"}

    def _auto(self):
        if self._automation is None:
            import pyautogui
            self._automation = pyautogui
        return self._automation

    def get_info(self):
        with self._lock:
            return {"config": deepcopy(self._config), "gestures": GESTURES, "profiles": PROFILES,
                    "postures": POSTURES, "cuda": self._cuda,
                    "model_ready": self._model_path.is_file(), "settings_path": str(self._store.path),
                    "platform": platform.system()}

    def get_state(self, frame_id=-1, gui_focused=None):
        with self._lock:
            if isinstance(gui_focused, bool):
                self._gui_focused = gui_focused
                if gui_focused and self._state["armed"]:
                    self._disarm()
            state = deepcopy(self._state)
            if state["frame_id"] == frame_id:
                state["frame"] = ""
            state["revision"] = self._revision
            state["posture_settings"] = deepcopy(self._config["postures"][self._config["posture"]])
            state["custom_gestures"] = deepcopy(self._config["custom_gestures"])
            return state

    def save_settings(self, config):
        try:
            config = validate_config(config)
            with self._lock:
                if self._capture:
                    raise ValueError("请等待手势录入结束后再修改配置")
                if config["cuda"] and not self._cuda["available"]:
                    raise ValueError(self._cuda["reason"])
                if self._state["running"] and any(config[k] != self._config[k] for k in ("camera", "cuda")):
                    raise ValueError("请先停止预览，再切换摄像头或 CUDA")
                # Editing a mapping always disarms input, including pending countdowns.
                self._disarm()
                self._config = self._store.save(config)
                self._revision += 1
            return {"ok": True, "config": deepcopy(config)}
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

    def prepare_model(self):
        with self._lock:
            if self._state["downloading"]:
                return {"ok": False, "error": "模型下载正在进行"}
            self._state["downloading"] = True
        try:
            download_model(self._model_path)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": f"下载失败：{exc}"}
        finally:
            with self._lock:
                self._state["downloading"] = False

    def start_preview(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"ok": False, "error": "摄像头正在运行或停止中"}
            if not self._model_path.is_file():
                return {"ok": False, "error": "请先点击「下载手部模型」"}
            if self._config["cuda"] and not self._cuda["available"]:
                return {"ok": False, "error": self._cuda["reason"]}
            self._stop.clear()
            self._disarm()
            self._state.update(running=True, error="", tuning="", capture="", gesture="等待手势")
            self._thread = threading.Thread(target=self._run, name="GestureCamera", daemon=True)
            self._thread.start()
        return {"ok": True}

    def stop_preview(self):
        with self._lock:
            self._stop.set()
            self._disarm()
            if self._state["tuning"]:
                self._store.save(self._config)
                self._state["tuning"] = ""
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)
        return {"ok": True}

    def _disarm(self):
        self._arm_at = 0
        self._state.update(armed=False, arming=0)

    def pause_output(self):
        with self._lock:
            self._disarm()
        return {"ok": True}

    def arm_output(self):
        with self._lock:
            if not self._state["running"] or self._stop.is_set():
                return {"ok": False, "error": "请先启动预览"}
            if self._state["tuning"] or self._capture:
                return {"ok": False, "error": "请先完成灵敏度调整或手势录入"}
            self._state["armed"] = False
            self._arm_at = time.monotonic() + 3
            self._state["arming"] = 3
        return {"ok": True}

    def set_tuning(self, field):
        if field not in {"", "gain_x", "gain_y", "smoothing"}:
            return {"ok": False, "error": "未知调整项目"}
        with self._lock:
            if field and (not self._state["running"] or self._capture):
                return {"ok": False, "error": "请启动预览并结束手势录入"}
            self._disarm()
            self._state["tuning"] = field
            self._revision += 1
            if not field:
                self._store.save(self._config)
        return {"ok": True}

    def center_on_hand(self):
        with self._lock:
            if not self._last_hand or time.monotonic() - self._last_hand_at > .5:
                return {"ok": False, "error": "请先启动预览，并将手保持在舒适的中立位置"}
            self._disarm()
            self._config["postures"][self._config["posture"]]["center"] = list(self._last_hand)
            self._store.save(self._config)
            self._revision += 1
        return {"ok": True}

    def capture_gesture(self, name):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 30:
            return {"ok": False, "error": "请输入 1–30 字的手势名称"}
        with self._lock:
            if not self._state["running"] or self._state["tuning"]:
                return {"ok": False, "error": "请启动预览并结束灵敏度调整"}
            if len(self._config["custom_gestures"]) >= 20:
                return {"ok": False, "error": "最多录入 20 个手势"}
            if self._capture:
                return {"ok": False, "error": "正在录入，请稍候"}
            self._disarm()
            self._capture = {"name": name.strip(), "start": time.monotonic() + 2, "samples": []}
            self._state["capture"] = "准备手势 · 2 秒后录入"
        return {"ok": True}

    def delete_gesture(self, ident):
        with self._lock:
            self._disarm()
            self._config["custom_gestures"] = [item for item in self._config["custom_gestures"] if item["id"] != ident]
            for bindings in self._config["bindings"].values():
                bindings.pop(ident, None)
            self._store.save(self._config)
            self._revision += 1
        return {"ok": True}

    def _enroll(self, landmarks, aspect, now):
        with self._lock:
            capture = self._capture
            if not capture:
                return False
            elapsed = now - capture["start"]
            if elapsed < 0:
                self._state["capture"] = f"准备手势 · {abs(elapsed):.1f} 秒"
            elif elapsed <= 1.5:
                if landmarks:
                    capture["samples"].append(pose_features(landmarks, aspect))
                self._state["capture"] = f"保持手势 · {max(0, 1.5 - elapsed):.1f} 秒"
            else:
                samples = np.asarray(capture["samples"])
                if len(samples) < 8 or np.max(np.std(samples, axis=0)) > .35:
                    self._state["capture"] = "录入失败：请将手放入镜头并保持稳定，再试一次"
                else:
                    ident = "pose_" + uuid.uuid4().hex[:12]
                    self._config["custom_gestures"].append({"id": ident, "name": capture["name"],
                                                           "features": samples.mean(axis=0).tolist()})
                    for bindings in self._config["bindings"].values():
                        bindings[ident] = "none"
                    self._store.save(self._config)
                    self._revision += 1
                    self._state["capture"] = "手势已录入，可在下方绑定操作"
                self._capture = None
            return True

    def _run(self):
        cap = None
        gate, mapper, tuner = GestureGate(), PointerMapper(), SensitivityTuner()
        last_tuning, previous_revision = "", -1
        frame_time = time.monotonic()
        preview_time = 0
        try:
            with self._lock:
                initial = deepcopy(self._config)
            if initial["cuda"] and not self._cuda["available"]:
                raise RuntimeError(self._cuda["reason"])
            gpu_mat = cv2.cuda_GpuMat() if initial["cuda"] else None
            with HandTracker(self._model_path, num_hands=1) as tracker:
                cap = cv2.VideoCapture(initial["camera"])
                if not cap.isOpened():
                    raise RuntimeError("摄像头无法打开：请检查编号、系统相机权限或是否被其他程序占用")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("摄像头画面读取失败，请停止后重试")
                    now = time.monotonic()
                    with self._lock:
                        config = deepcopy(self._config)
                        tuning = self._state["tuning"]
                        if previous_revision != self._revision:
                            gate.reset()
                            mapper.reset()
                            tuner.anchor = None
                            previous_revision = self._revision
                        if self._arm_at:
                            self._state["arming"] = max(0, round(self._arm_at - now, 1))
                            if now >= self._arm_at:
                                self._state["armed"] = True
                                self._arm_at = 0
                                gate.reset()
                                gate.latched = True  # Require a release after enabling control.
                        if platform.system() == "Windows" and ctypes.windll.user32.GetAsyncKeyState(0x77) & 0x8000:
                            self._disarm()
                        armed = self._state["armed"] and not self._gui_focused
                    frame = cv2.flip(frame, 1)
                    if gpu_mat is not None:
                        gpu_mat.upload(frame)
                        rgb = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2RGB).download()
                    else:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = tracker.process(rgb)
                    landmarks = result.hand_landmarks[0] if result.hand_landmarks else []
                    aspect = frame.shape[1] / frame.shape[0]
                    enrolling = self._enroll(landmarks, aspect, now)
                    gesture = None
                    if landmarks:
                        gesture, pinching = recognize(landmarks, aspect, config["pinch_threshold"], config["custom_gestures"])
                        draw_landmarks(frame, landmarks)
                        wrist = (landmarks[0].x, landmarks[0].y)
                        with self._lock:
                            self._last_hand, self._last_hand_at = wrist, now
                        if tuning:
                            if tuning != last_tuning:
                                tuner.anchor = None
                            posture = config["postures"][config["posture"]]
                            point = (landmarks[8].x, landmarks[8].y)
                            updated = tuner.update(point, pinching, posture[tuning], tuning)
                            with self._lock:
                                # Do not apply a stale camera frame to a changed posture.
                                if self._revision == previous_revision:
                                    self._config["postures"][config["posture"]][tuning] = updated
                        elif armed and not enrolling:
                            action = config["bindings"][config["mode"]].get(gesture, "none")
                            event = gate.update(gesture if action != "none" else None, now, config["hold_ms"] / 1000)
                            automation = self._auto()
                            # Recheck under the lock: a concurrent Pause/Save must win.
                            with self._lock:
                                if self._state["armed"] and not self._gui_focused and not self._stop.is_set() and self._revision == previous_revision:
                                    if config["mode"] == "mouse":
                                        screen = automation.size()
                                        x, y = mapper.map(*wrist, config["postures"][config["posture"]], *screen)
                                        automation.moveTo(x, y, _pause=False)
                                    if event:
                                        dispatch(action, automation)
                                        self._state["last_action"] = action
                        else:
                            gate.reset()
                            mapper.reset()
                    else:
                        gate.update(None, now)
                        mapper.reset()
                        tuner.anchor = None
                    last_tuning = tuning
                    label = GESTURES.get(gesture) or next((g["name"] for g in config["custom_gestures"] if g["id"] == gesture), "未匹配手势")
                    with self._lock:
                        self._state.update(fps=round(1 / max(now - frame_time, .001)),
                                           gesture=label if landmarks else "未检测到手部",
                                           landmarks=[{"x": p.x, "y": p.y} for p in landmarks],
                                           preprocessing="CUDA" if gpu_mat is not None else "CPU")
                        if now - preview_time >= .08:
                            encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                            if encoded:
                                self._state["frame"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")
                                self._state["frame_id"] += 1
                            preview_time = now
                    frame_time = now
        except Exception as exc:
            with self._lock:
                self._state["error"] = ("已触发鼠标角落紧急停止。" if type(exc).__name__ == "FailSafeException" else str(exc))
        finally:
            if cap is not None:
                cap.release()
            with self._lock:
                self._disarm()
                self._capture = None
                self._last_hand = None
                self._state.update(running=False, tuning="", landmarks=[], capture="", fps=0)


class Bridge:
    """Only explicitly exposed public methods can be called by the frontend."""
    def __init__(self, controller):
        self._controller = controller

    def get_info(self):
        return self._controller.get_info()

    def get_state(self, frame_id=-1, gui_focused=None):
        return self._controller.get_state(frame_id, gui_focused)

    def save_settings(self, config):
        return self._controller.save_settings(config)

    def start_preview(self):
        return self._controller.start_preview()

    def stop_preview(self):
        return self._controller.stop_preview()

    def arm_output(self):
        return self._controller.arm_output()

    def pause_output(self):
        return self._controller.pause_output()

    def set_tuning(self, field):
        return self._controller.set_tuning(field)

    def center_on_hand(self):
        return self._controller.center_on_hand()

    def capture_gesture(self, name):
        return self._controller.capture_gesture(name)

    def delete_gesture(self, ident):
        return self._controller.delete_gesture(ident)

    def prepare_model(self):
        return self._controller.prepare_model()
