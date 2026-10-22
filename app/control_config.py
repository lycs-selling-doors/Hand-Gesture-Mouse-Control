"""Validated, portable settings for the desktop controller."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "user-settings.json"
GESTURES = {
    "thumb_index": "拇指 + 食指捏合", "thumb_middle": "拇指 + 中指捏合",
    "thumb_ring": "拇指 + 无名指捏合", "thumb_pinky": "拇指 + 小指捏合",
    "index_middle": "食指 + 中指并拢", "open_palm": "张开手掌",
    "fist": "握拳", "victory": "剪刀手",
}
PROFILES = {"mouse": "鼠标模式", "ppt": "PPT 模式", "media": "媒体模式", "custom": "自定义模式"}
POSTURES = {"seated": "坐姿", "standing": "站姿", "reclined": "倚靠", "custom": "自定义"}
KEYS = set("abcdefghijklmnopqrstuvwxyz0123456789") | {
    "ctrl", "alt", "shift", "win", "enter", "esc", "space", "tab", "backspace",
    "delete", "insert", "home", "end", "pageup", "pagedown", "left", "right", "up", "down",
    "playpause", "nexttrack", "prevtrack", "volumeup", "volumedown", "volumemute",
    "printscreen", "pause", "capslock", "numlock", "scrolllock", "decimal", "add", "subtract",
    "multiply", "divide", ",", ".", "/", ";", "'", "[", "]", "-", "=", "`", "\\",
} | {f"f{i}" for i in range(1, 25)}


def default_config():
    bindings = {mode: {gesture: "none" for gesture in GESTURES} for mode in PROFILES}
    bindings["mouse"].update(thumb_index="click:left", index_middle="click:right",
                             thumb_middle="scroll:-3", thumb_ring="scroll:3", thumb_pinky="doubleclick")
    bindings["ppt"].update(thumb_index="key:right", thumb_middle="key:left",
                           thumb_ring="key:f5", thumb_pinky="key:esc", victory="key:b")
    bindings["media"].update(thumb_index="key:playpause", thumb_middle="key:nexttrack",
                             thumb_ring="key:volumeup", thumb_pinky="key:volumedown", fist="key:volumemute")
    posture = {"gain_x": 2.2, "gain_y": 2.8, "smoothing": 0.4, "center": [0.5, 0.55]}
    return {"version": 1, "mode": "mouse", "posture": "seated", "camera": 0,
            "cuda": False, "pinch_threshold": 0.32, "hold_ms": 200,
            "postures": {name: deepcopy(posture) for name in POSTURES},
            "bindings": bindings, "custom_gestures": []}


def number(value, low, high, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} 必须是有限数字")
    if not low <= value <= high:
        raise ValueError(f"{label} 应在 {low}–{high} 之间")
    return float(value)


def validate_action(action):
    if not isinstance(action, str) or len(action) > 80:
        raise ValueError("无效按键映射")
    action = action.strip().lower()
    if action in {"none", "click:left", "click:right", "doubleclick", "scroll:3", "scroll:-3"}:
        return action
    kind, _, value = action.partition(":")
    keys = value.split("+")
    if kind not in {"key", "hotkey"} or not 1 <= len(keys) <= 5 or any(k not in KEYS for k in keys):
        raise ValueError("使用 key:right 或 hotkey:ctrl+shift+p 等有效按键")
    if "f8" in keys:
        raise ValueError("F8 保留为紧急暂停键")
    if kind == "key" and len(keys) != 1:
        raise ValueError("组合键请使用 hotkey: 前缀")
    return action


def validate_config(value):
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("不支持的配置格式")
    config = default_config()
    for field, choices in (("mode", PROFILES), ("posture", POSTURES)):
        if value.get(field) not in choices:
            raise ValueError(f"无效 {field}")
        config[field] = value[field]
    camera = value.get("camera")
    if isinstance(camera, bool) or not isinstance(camera, int) or not 0 <= camera <= 10:
        raise ValueError("摄像头编号应为 0–10")
    config["camera"] = camera
    if not isinstance(value.get("cuda"), bool):
        raise ValueError("CUDA 选项必须是布尔值")
    config["cuda"] = value["cuda"]
    config["pinch_threshold"] = number(value.get("pinch_threshold"), .15, .65, "捏合阈值")
    config["hold_ms"] = int(number(value.get("hold_ms"), 100, 1000, "保持时间"))
    for name in POSTURES:
        data = value.get("postures", {}).get(name, {})
        center = data.get("center", [])
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError("中心点需要两个坐标")
        config["postures"][name] = {
            "gain_x": number(data.get("gain_x"), .5, 5, "水平灵敏度"),
            "gain_y": number(data.get("gain_y"), .5, 5, "垂直灵敏度"),
            "smoothing": number(data.get("smoothing"), .1, .9, "响应速度"),
            "center": [number(c, 0, 1, "中心点") for c in center],
        }
    customs = value.get("custom_gestures", [])
    if not isinstance(customs, list) or len(customs) > 20:
        raise ValueError("最多保存 20 个自定义手势")
    names = set(GESTURES)
    for item in customs:
        if not isinstance(item, dict):
            raise ValueError("无效自定义手势")
        ident, label, features = item.get("id"), item.get("name"), item.get("features")
        if not isinstance(ident, str) or not ident.startswith("pose_") or len(ident) > 50 or ident in names:
            raise ValueError("自定义手势 ID 无效或重复")
        if not isinstance(label, str) or not 1 <= len(label.strip()) <= 30:
            raise ValueError("手势名称需要 1–30 个字符")
        if not isinstance(features, list) or len(features) != 25:
            raise ValueError("自定义手势特征无效")
        config["custom_gestures"].append({"id": ident, "name": label.strip(),
                                        "features": [number(f, 0, 20, "手势特征") for f in features]})
        names.add(ident)
    for mode in PROFILES:
        bindings = value.get("bindings", {}).get(mode)
        if not isinstance(bindings, dict) or any(key not in names for key in bindings):
            raise ValueError("绑定引用了未知手势")
        config["bindings"][mode] = {gesture: validate_action(bindings.get(gesture, "none")) for gesture in names}
    return config


class ConfigStore:
    def __init__(self, path=SETTINGS_PATH):
        self.path = Path(path)
        self.error = ""

    def load(self):
        try:
            return validate_config(json.loads(self.path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return default_config()
        except (ValueError, TypeError, AttributeError) as exc:
            self.error = f"配置读取失败，暂用默认值：{exc}"
            return default_config()

    def save(self, config):
        config = validate_config(config)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.error and self.path.exists():
            backup = self.path.with_suffix(".invalid.json")
            if not backup.exists():
                backup.write_bytes(self.path.read_bytes())
        path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             suffix=".tmp", delete=False) as stream:
                path = Path(stream.name)
                json.dump(config, stream, ensure_ascii=False, indent=2)
            path.replace(self.path)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
        self.error = ""
        return config
