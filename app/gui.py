"""Launch the local pywebview desktop control panel: python -m app.gui."""

import argparse
import json
from pathlib import Path
import time

from .desktop_controller import Bridge, DesktopController

ASSETS = Path(__file__).resolve().parent / "frontend"


def build_html():
    return (ASSETS.joinpath("index.html").read_text(encoding="utf-8")
            .replace("/*INLINE_STYLE*/", ASSETS.joinpath("style.css").read_text(encoding="utf-8"))
            .replace("/*INLINE_SCRIPT*/", ASSETS.joinpath("app.js").read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    import webview

    controller = DesktopController()
    window = webview.create_window("Gesture Desk · 手势控制台", html=build_html(),
                                   js_api=Bridge(controller), width=1360, height=940,
                                   min_size=(1080, 740), background_color="#f5f6f8")
    def on_closed():
        controller.stop_preview()

    window.events.closed += on_closed

    def smoke_test():
        # Exercise the real WebView2 bridge without camera capture or input output.
        try:
            window.events.loaded.wait(timeout=20)
            deadline = time.monotonic() + 20
            result = None
            while time.monotonic() < deadline and result is None:
                result = window.evaluate_js("window.desktopSmokeResult || null")
                time.sleep(.1)
            if result is None:
                raise RuntimeError("JavaScript bridge initialization timed out")
            args.smoke_test.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            args.smoke_test.write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        finally:
            window.destroy()

    if args.smoke_test:
        webview.start(smoke_test, gui="edgechromium", debug=False)
    else:
        webview.start(gui="edgechromium" if __import__("sys").platform == "win32" else None, debug=False)


if __name__ == "__main__":
    main()
