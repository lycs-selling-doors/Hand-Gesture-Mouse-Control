"""Desktop regression tests. Hardware and OS input are mocked."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from app.control_config import ConfigStore, default_config, validate_action, validate_config
from app.desktop_controller import DesktopController, cuda_capability, dispatch
from app.gesture_logic import (GestureGate, MovementGate, PointerMapper, SensitivityTuner,
                               is_move_gesture, pose_features, recognize)


def hand():
    points = [SimpleNamespace(x=.5, y=.7, z=0.) for _ in range(21)]
    coords = {0: (.5, .9), 4: (.23, .57), 5: (.4, .68), 6: (.4, .45), 8: (.4, .2),
              9: (.5, .65), 10: (.5, .4), 12: (.5, .15), 13: (.6, .67), 14: (.6, .45),
              16: (.6, .22), 17: (.7, .72), 18: (.7, .52), 20: (.7, .32)}
    for index, (x, y) in coords.items():
        points[index] = SimpleNamespace(x=x, y=y, z=0.)
    return points


class ConfigurationTests(unittest.TestCase):
    def test_defaults_round_trip_and_presets_are_independent(self):
        config = default_config()
        self.assertEqual(validate_config(config), config)
        config["postures"]["standing"]["gain_x"] = 4
        self.assertEqual(config["postures"]["seated"]["gain_x"], 2.2)
        self.assertEqual(config["bindings"]["ppt"]["thumb_index"], "key:right")
        self.assertEqual(config["bindings"]["media"]["thumb_index"], "key:playpause")

    def test_rejects_nonfinite_and_out_of_range_sensitivity(self):
        for invalid in (float("nan"), float("inf"), True, 0, 6):
            config = default_config()
            config["postures"]["seated"]["gain_x"] = invalid
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_rejects_unknown_keys_and_reserved_emergency_key(self):
        for action in ("shell:calc", "hotkey:ctrl+unknown", "key:f8", "key:ctrl+p", "hotkey:"):
            with self.assertRaises(ValueError):
                validate_action(action)
        self.assertEqual(validate_action("hotkey:ctrl+shift+p"), "hotkey:ctrl+shift+p")

    def test_old_movement_right_click_is_removed_without_resetting_user_settings(self):
        config = default_config(); config.pop("pointer_anchor")
        config["bindings"]["mouse"].update(index_middle="click:right", fist="click:left", thumb_index="none")
        config["postures"]["seated"]["gain_y"] = 3.4
        migrated = validate_config(config)
        self.assertEqual(migrated["bindings"]["mouse"]["index_middle"], "none")
        self.assertEqual(migrated["bindings"]["mouse"]["fist"], "click:left")
        self.assertEqual(migrated["postures"]["seated"]["gain_y"], 3.4)
        self.assertEqual(migrated["pointer_anchor"], "index_tip")

    def test_custom_pose_validation_and_binding(self):
        config = default_config()
        config["custom_gestures"] = [{"id": "pose_test", "name": "测试手型", "features": pose_features(hand())}]
        config["bindings"]["custom"]["pose_test"] = "hotkey:ctrl+enter"
        self.assertEqual(validate_config(config)["bindings"]["custom"]["pose_test"], "hotkey:ctrl+enter")
        config["custom_gestures"].append(deepcopy(config["custom_gestures"][0]))
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_invalid_save_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConfigStore(Path(directory) / "settings.json")
            store.save(default_config())
            original = store.path.read_bytes()
            invalid = default_config(); invalid["camera"] = -1
            with self.assertRaises(ValueError):
                store.save(invalid)
            self.assertEqual(store.path.read_bytes(), original)

    def test_corrupt_settings_have_visible_error_and_backup_on_save(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConfigStore(Path(directory) / "settings.json")
            store.path.write_text("corrupt", encoding="utf-8")
            self.assertEqual(store.load(), default_config())
            self.assertTrue(store.error)
            store.save(default_config())
            self.assertEqual(store.path.with_suffix(".invalid.json").read_text(), "corrupt")


class GestureTests(unittest.TestCase):
    def test_only_extended_together_fingers_enable_movement(self):
        landmarks = hand()
        self.assertFalse(is_move_gesture(landmarks))
        landmarks[12] = SimpleNamespace(x=.42, y=.2, z=0.)
        self.assertTrue(is_move_gesture(landmarks))
        landmarks[8] = SimpleNamespace(x=.5, y=.8, z=0.)
        landmarks[12] = SimpleNamespace(x=.5, y=.8, z=0.)
        self.assertFalse(is_move_gesture(landmarks))

    def test_pinches_never_enable_movement(self):
        landmarks = hand(); landmarks[12] = SimpleNamespace(x=.42, y=.2, z=0.)
        landmarks[4] = deepcopy(landmarks[8])
        self.assertFalse(is_move_gesture(landmarks))

    def test_movement_has_short_activation_and_immediate_release(self):
        gate = MovementGate()
        self.assertFalse(gate.update(True, 0))
        self.assertFalse(gate.update(True, .05))
        self.assertTrue(gate.update(True, .11))
        self.assertFalse(gate.update(False, .12))
        self.assertFalse(gate.update(True, .13))

    def test_open_palm_and_pinches_are_resolution_independent(self):
        landmarks = hand()
        self.assertEqual(recognize(landmarks, 1, .32)[0], "open_palm")
        landmarks[8] = deepcopy(landmarks[4])
        self.assertEqual(recognize(landmarks, 1, .32)[0], "thumb_index")
        for point in landmarks:
            point.x = .2 + point.x * .5; point.y = .1 + point.y * .5
        self.assertEqual(recognize(landmarks, 1, .32)[0], "thumb_index")

    def test_fist_is_not_mistaken_for_multiple_pinches(self):
        landmarks = hand()
        for i in (8, 12, 16, 20):
            landmarks[i] = SimpleNamespace(x=.5, y=.82, z=0.)
        self.assertEqual(recognize(landmarks, 1, .32)[0], "fist")

    def test_recorded_pose_matches_translated_and_scaled_hand(self):
        landmarks = hand()
        custom = {"id": "pose_me", "features": pose_features(landmarks)}
        for point in landmarks:
            point.x = .1 + point.x * .7; point.y = .15 + point.y * .7
        self.assertEqual(recognize(landmarks, 1, .32, [custom])[0], "pose_me")

    def test_hold_fires_once_and_requires_neutral_release(self):
        gate = GestureGate()
        self.assertIsNone(gate.update("pinch", 0))
        self.assertEqual(gate.update("pinch", .21), "pinch")
        self.assertIsNone(gate.update("pinch", 1))
        self.assertIsNone(gate.update("other", 2))
        gate.update(None, 2.1); gate.update(None, 2.25)
        gate.update("other", 2.3)
        self.assertEqual(gate.update("other", 2.51), "other")

    def test_short_noise_does_not_trigger_or_rearm(self):
        gate = GestureGate()
        gate.update("a", 0); self.assertIsNone(gate.update("b", .1))
        gate.update("b", .31)
        gate.update(None, .4); self.assertIsNone(gate.update("b", .45))

    def test_sensitivity_changes_travel_without_changing_neutral_center(self):
        posture = default_config()["postures"]["seated"]
        center = posture["center"]
        x, y = PointerMapper().map(*center, posture, 1920, 1080)
        self.assertLess(abs(x - 960), 2); self.assertLess(abs(y - 540), 2)
        low = PointerMapper().map(center[0] + .05, center[1], posture, 1920, 1080)[0]
        posture["gain_x"] = 4
        high = PointerMapper().map(center[0] + .05, center[1], posture, 1920, 1080)[0]
        self.assertGreater(high, low)

    def test_pointer_is_clamped_away_from_emergency_corners(self):
        posture = default_config()["postures"]["seated"]
        for x, y in [(-100, -100), (100, 100)]:
            px, py = PointerMapper().map(x, y, posture, 1920, 1080)
            self.assertTrue(0 < px < 1919); self.assertTrue(0 < py < 1079)

    def test_pinch_tuning_has_anchor_release_and_limits(self):
        tuner = SensitivityTuner()
        self.assertEqual(tuner.update((.5, .5), True, 2, "gain_x"), 2)
        self.assertEqual(tuner.update((.7, .5), True, 2, "gain_x"), 3.4)
        tuner.update((.7, .5), False, 3.4, "gain_x")
        self.assertEqual(tuner.update((.1, .5), True, 3.4, "gain_x"), 3.4)
        self.assertEqual(tuner.update((1, .5), True, 3.4, "gain_x"), 5)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ConfigStore(Path(self.directory.name) / "settings.json")
        self.automation = MagicMock()
        self.automation.size.return_value = (1920, 1080)
        self.automation.FAILSAFE = True
        self.patch = patch("app.desktop_controller.cuda_capability", return_value={"available": False, "reason": "No CUDA", "devices": 0})
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.controller = DesktopController(self.store, self.automation)

    def test_preview_missing_model_does_not_open_camera(self):
        self.controller._model_path = Path(self.directory.name) / "missing.task"
        with patch("app.desktop_controller.cv2.VideoCapture") as camera:
            self.assertFalse(self.controller.start_preview()["ok"])
            camera.assert_not_called()

    def test_unsupported_cuda_cannot_be_enabled(self):
        config = default_config(); config["cuda"] = True
        self.assertFalse(self.controller.save_settings(config)["ok"])
        self.assertFalse(self.controller.get_info()["config"]["cuda"])

    def test_settings_cannot_overwrite_a_concurrent_enrollment(self):
        self.controller._capture = {"name": "recording"}
        self.assertFalse(self.controller.save_settings(default_config())["ok"])

    def test_stop_saves_in_progress_sensitivity_tuning(self):
        self.controller._state["tuning"] = "gain_x"
        self.controller._config["postures"]["seated"]["gain_x"] = 3.6
        self.controller.stop_preview()
        self.assertEqual(self.store.load()["postures"]["seated"]["gain_x"], 3.6)

    def test_save_cancels_pending_and_active_output(self):
        self.controller._state["armed"] = True; self.controller._arm_at = 9999
        self.assertTrue(self.controller.save_settings(default_config())["ok"])
        self.assertFalse(self.controller.get_state()["armed"])
        self.assertEqual(self.controller._arm_at, 0)

    def test_returning_to_gui_pauses_output(self):
        self.controller._state["armed"] = True
        self.assertFalse(self.controller.get_state(-1, True)["armed"])

    def test_no_arm_during_tuning_or_enrollment(self):
        self.controller._state.update(running=True, tuning="gain_x")
        self.assertFalse(self.controller.arm_output()["ok"])
        self.controller._state["tuning"] = ""
        self.controller._capture = {"name": "recording"}
        self.assertFalse(self.controller.arm_output()["ok"])

    def test_center_calibration_requires_fresh_hand_and_saves_per_posture(self):
        self.assertFalse(self.controller.center_on_hand()["ok"])
        with patch("app.desktop_controller.time.monotonic", return_value=10):
            self.controller._last_hand = (.3, .6); self.controller._last_hand_at = 10
            self.assertTrue(self.controller.center_on_hand()["ok"])
        saved = self.store.load()
        self.assertEqual(saved["postures"]["seated"]["center"], [.3, .6])
        self.assertEqual(saved["postures"]["standing"]["center"], [.5, .55])

    def test_custom_enrollment_saves_signature_and_all_mode_bindings(self):
        self.controller._capture = {"name": "我的手势", "start": 0, "samples": [pose_features(hand()) for _ in range(10)]}
        self.controller._enroll(hand(), 1, 2)
        config = self.store.load(); custom = config["custom_gestures"][0]
        self.assertEqual(custom["name"], "我的手势")
        self.assertEqual(len(custom["features"]), 25)
        for bindings in config["bindings"].values():
            self.assertEqual(bindings[custom["id"]], "none")

    def test_preview_worker_never_dispatches_input(self):
        with patch("app.desktop_controller.HandTracker") as tracker, patch("app.desktop_controller.cv2.VideoCapture") as camera:
            tracker.return_value.__enter__.return_value.process.return_value = SimpleNamespace(hand_landmarks=[hand()])
            camera.return_value.isOpened.return_value = True
            camera.return_value.read.side_effect = [(True, np.zeros((240, 320, 3), np.uint8)), (False, None)]
            self.controller._run()
            self.automation.moveTo.assert_not_called(); self.automation.press.assert_not_called()
            camera.return_value.release.assert_called_once()
            tracker.return_value.__exit__.assert_called_once()
            self.assertFalse(self.controller.get_state()["running"])

    def test_keyboard_mode_never_moves_mouse(self):
        self.controller._config["mode"] = "ppt"
        self.controller._state["armed"] = True; self.controller._gui_focused = False
        with patch("app.desktop_controller.HandTracker") as tracker, patch("app.desktop_controller.cv2.VideoCapture") as camera:
            tracker.return_value.__enter__.return_value.process.return_value = SimpleNamespace(hand_landmarks=[hand()])
            camera.return_value.isOpened.return_value = True
            camera.return_value.read.side_effect = [(True, np.zeros((240, 320, 3), np.uint8)), (False, None)]
            self.controller._run()
            self.automation.moveTo.assert_not_called()

    def test_worker_outputs_one_ppt_key_for_a_held_pinch(self):
        landmarks = hand(); landmarks[8] = deepcopy(landmarks[4])
        self.controller._config.update(mode="ppt", hold_ms=100)
        self.controller._state["armed"] = True; self.controller._gui_focused = False
        frame = np.zeros((240, 320, 3), np.uint8)
        clock = iter(i * .12 for i in range(100))
        with patch("app.desktop_controller.HandTracker") as tracker, patch("app.desktop_controller.cv2.VideoCapture") as camera:
            tracker.return_value.__enter__.return_value.process.return_value = SimpleNamespace(hand_landmarks=[landmarks])
            camera.return_value.isOpened.return_value = True
            camera.return_value.read.side_effect = [(True, frame)] * 6 + [(False, None)]
            with patch("app.desktop_controller.time.monotonic", side_effect=lambda: next(clock)):
                self.controller._run()
        self.automation.press.assert_called_once_with("right", _pause=False)
        self.automation.moveTo.assert_not_called()

    def test_worker_moves_from_index_tip_only_while_movement_pose_is_held(self):
        held = hand(); held[12] = SimpleNamespace(x=.42, y=.2, z=0.)
        self.controller._state["armed"] = True; self.controller._gui_focused = False
        # Even an overlapping custom pose with a click mapping cannot click during movement.
        self.controller._config["custom_gestures"] = [{"id": "pose_move", "name": "重叠手型",
                                                      "features": pose_features(held, 320 / 240)}]
        self.controller._config["bindings"]["mouse"]["pose_move"] = "click:right"
        frame = np.zeros((240, 320, 3), np.uint8)
        clock = iter(i * .12 for i in range(100))
        with patch("app.desktop_controller.HandTracker") as tracker, patch("app.desktop_controller.cv2.VideoCapture") as camera:
            tracker.return_value.__enter__.return_value.process.side_effect = [
                SimpleNamespace(hand_landmarks=[held]), SimpleNamespace(hand_landmarks=[held]),
                SimpleNamespace(hand_landmarks=[hand()]), SimpleNamespace(hand_landmarks=[])]
            camera.return_value.isOpened.return_value = True
            camera.return_value.read.side_effect = [(True, frame)] * 4 + [(False, None)]
            with patch("app.desktop_controller.time.monotonic", side_effect=lambda: next(clock)):
                self.controller._run()
        expected = PointerMapper().map(held[8].x, held[8].y,
                                      self.controller._config["postures"]["seated"], 1920, 1080)
        self.automation.moveTo.assert_called_once_with(*expected, _pause=False)
        self.automation.click.assert_not_called()

    def test_shortcuts_release_modifiers_even_when_keydown_fails(self):
        self.automation.keyDown.side_effect = [None, RuntimeError("input failed")]
        with self.assertRaisesRegex(RuntimeError, "input failed"):
            dispatch("hotkey:ctrl+p", self.automation)
        self.assertEqual([c.args[0] for c in self.automation.keyUp.call_args_list], ["p", "ctrl"])
        self.assertTrue(self.automation.FAILSAFE)

    def test_ppt_and_media_dispatch_keys(self):
        dispatch("key:right", self.automation); dispatch("key:playpause", self.automation)
        self.assertEqual([c.args[0] for c in self.automation.press.call_args_list], ["right", "playpause"])


class CudaTests(unittest.TestCase):
    def test_device_count_alone_is_insufficient(self):
        with patch("app.desktop_controller.cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
            with patch("app.desktop_controller.cv2.cuda.cvtColor", create=True, side_effect=RuntimeError("missing backend")):
                self.assertFalse(cuda_capability()["available"])

    def test_cuda_success_requires_a_real_conversion_probe(self):
        with patch("app.desktop_controller.cv2.cuda.getCudaEnabledDeviceCount", return_value=1):
            with patch("app.desktop_controller.cv2.cuda_GpuMat") as mat:
                with patch("app.desktop_controller.cv2.cuda.cvtColor", create=True) as conversion:
                    self.assertTrue(cuda_capability()["available"])
                    mat.return_value.upload.assert_called_once()
                    conversion.return_value.download.assert_called_once()


if __name__ == "__main__":
    unittest.main()
