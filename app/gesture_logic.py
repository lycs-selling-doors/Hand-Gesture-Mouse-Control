"""Pure geometry, posture mapping, gesture recognition and release-to-rearm logic."""

from math import dist, sqrt


def position(point, aspect=1):
    return (point.x * aspect, point.y, getattr(point, "z", 0) * aspect)


def pose_features(landmarks, aspect=1):
    points = [position(point, aspect) for point in landmarks]
    scale = max(dist(points[5], points[17]), .02)
    pairs = [(0, i) for i in range(1, 21)] + [(4, i) for i in (8, 12, 16, 20)] + [(8, 12)]
    return [dist(points[a], points[b]) / scale for a, b in pairs]


def is_move_gesture(landmarks, aspect=1, threshold=.32):
    """Both fingers extended and together; pinches must never move the pointer."""
    points = [position(point, aspect) for point in landmarks]
    scale = max(dist(points[5], points[17]), .02)
    extended = all(dist(points[0], points[tip]) > dist(points[0], points[pip]) * 1.22
                   for tip, pip in ((8, 6), (12, 10)))
    separated_from_thumb = all(dist(points[4], points[tip]) / scale >= threshold
                               for tip in (8, 12, 16, 20))
    return extended and separated_from_thumb and dist(points[8], points[12]) / scale < threshold


class MovementGate:
    """Brief stable activation, but immediate release with no trailing motion."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.since = None

    def update(self, held, now):
        if not held:
            self.reset()
            return False
        if self.since is None:
            self.since = now
        return now - self.since >= .1


def recognize(landmarks, aspect, threshold, custom_gestures=()):
    # Distances normalized by palm width adapt to camera distance and resolution.
    points = [position(point, aspect) for point in landmarks]
    scale = max(dist(points[5], points[17]), .02)
    pinches = [("thumb_index", 4, 8), ("thumb_middle", 4, 12), ("thumb_ring", 4, 16),
               ("thumb_pinky", 4, 20)]
    distances = [(name, dist(points[a], points[b]) / scale) for name, a, b in pinches]
    closest, distance = min(distances, key=lambda item: item[1])
    pinching = dict(distances)["thumb_index"] < threshold
    if custom_gestures:
        features = pose_features(landmarks, aspect)
        errors = [(sqrt(sum((a - b) ** 2 for a, b in zip(features, item["features"])) / 25), item["id"])
                  for item in custom_gestures]
        error, ident = min(errors)
        if error < .16:
            return ident, pinching
    extended = [dist(points[0], points[tip]) > dist(points[0], points[pip]) * 1.22
                for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
    if not any(extended):
        return "fist", pinching
    if distance < threshold:
        return closest, pinching
    if all(extended):
        if dist(points[8], points[12]) / scale < threshold * .8:
            return "index_middle", pinching
        return "open_palm", pinching
    if extended == [True, True, False, False]:
        if dist(points[8], points[12]) / scale < threshold:
            return "index_middle", pinching
        return "victory", pinching
    return None, pinching


class GestureGate:
    """One action per hold; neutral release is required before the next action."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.candidate = None
        self.since = 0
        self.latched = False
        self.neutral_since = None

    def update(self, gesture, now, hold=.2):
        if gesture is None:
            if self.neutral_since is None:
                self.neutral_since = now
            if now - self.neutral_since >= .12:
                self.reset()
            return None
        self.neutral_since = None
        if self.latched:
            return None
        if gesture != self.candidate:
            self.candidate, self.since = gesture, now
        elif now - self.since >= hold:
            self.latched = True
            return gesture
        return None


class PointerMapper:
    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def map(self, x, y, posture, width, height):
        target = [min(.998, max(.002, .5 + (point - center) * gain))
                  for point, center, gain in zip((x, y), posture["center"],
                                                  (posture["gain_x"], posture["gain_y"]))]
        if self.previous is not None:
            alpha = posture["smoothing"]
            target = [old if abs(new - old) < .002 else old + alpha * (new - old)
                      for old, new in zip(self.previous, target)]
        self.previous = target
        return round(target[0] * (width - 1)), round(target[1] * (height - 1))


class SensitivityTuner:
    """Pinch then drag: horizontal motion changes gain, vertical changes smoothing."""
    def __init__(self):
        self.anchor = None

    def update(self, point, pinching, value, field):
        if not pinching:
            self.anchor = None
            return value
        if self.anchor is None:
            self.anchor = (point, value)
        origin, start = self.anchor
        if field == "smoothing":
            return round(min(.9, max(.1, start - (point[1] - origin[1]) * 1.4)), 2)
        return round(min(5, max(.5, start + (point[0] - origin[0]) * 7)), 2)
