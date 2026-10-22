"""Pure geometry, posture mapping, gesture recognition and release-to-rearm logic."""

from math import acos, cos, dist, radians, sqrt

AMBIGUOUS = "__ambiguous__"


def position(point, aspect=1):
    return (point.x * aspect, point.y, getattr(point, "z", 0) * aspect)


def pose_features(landmarks, aspect=1):
    points = [position(point, aspect) for point in landmarks]
    scale = max(dist(points[5], points[17]), .02)
    pairs = [(0, i) for i in range(1, 21)] + [(4, i) for i in (8, 12, 16, 20)] + [(8, 12)]
    return [dist(points[a], points[b]) / scale for a, b in pairs]


def direction_cosine(a, b):
    length = sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    if length < 1e-8:
        return -1.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / length))


def vector(start, end):
    return tuple(b - a for a, b in zip(start, end))


def builtin_gesture(landmarks, aspect, pinch_threshold):
    """Exclusive built-in classification with an explicit neutral ambiguity band."""
    points = [position(point, aspect) for point in landmarks]
    scale = max(dist(points[5], points[17]), .02)
    extended, folded = [], []
    for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        straight = direction_cosine(vector(points[mcp], points[pip]), vector(points[pip], points[tip])) >= cos(radians(35))
        radial = dist(points[0], points[tip]) / max(dist(points[0], points[pip]), .02)
        extended.append(straight and radial > 1.22)
        folded.append(radial < 1.05 and dist(points[0], points[tip]) <= dist(points[0], points[mcp]) * 1.3)

    contacts = sorted((dist(points[4], points[tip]) / scale, name) for name, tip in
                      (("thumb_index", 8), ("thumb_middle", 12), ("thumb_ring", 16), ("thumb_pinky", 20)))
    if all(folded):
        return "fist"
    if contacts[0][0] < pinch_threshold:
        # A thumb close to several fingertips is ambiguous, not a reliable pinch.
        if contacts[1][0] - contacts[0][0] < .12:
            return AMBIGUOUS
        return contacts[0][1]

    if extended[0] and extended[1]:
        gap = dist(points[8], points[12]) / scale
        angle = acos(direction_cosine(vector(points[5], points[8]), vector(points[9], points[12])))
        # Movement thresholds are independent of the adjustable thumb-pinch threshold.
        if gap <= .28 and angle <= radians(16):
            return "index_middle"
        if not extended[2] and not extended[3] and gap >= .65 and angle >= radians(22):
            return "victory"
        # Do not reinterpret intermediate two-finger poses as a fist or a pinch.
        if not all(extended):
            return AMBIGUOUS
    if all(extended):
        return "open_palm"
    return None


def is_move_gesture(landmarks, aspect=1, threshold=.32):
    return builtin_gesture(landmarks, aspect, threshold) == "index_middle"


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
    builtin = builtin_gesture(landmarks, aspect, threshold)
    pinching = builtin == "thumb_index"
    if builtin == AMBIGUOUS:
        return None, False
    if builtin in {"index_middle", "victory"} or (builtin and builtin.startswith("thumb_")):
        return builtin, pinching
    if custom_gestures:
        features = pose_features(landmarks, aspect)
        errors = [(sqrt(sum((a - b) ** 2 for a, b in zip(features, item["features"])) / 25), item["id"])
                  for item in custom_gestures]
        error, ident = min(errors)
        if error < .16:
            return ident, pinching
    return builtin, pinching


class MovementTransitionGuard:
    """Stopping movement is immediate; adjacent key/click gestures wait for settling."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_movement = float("-inf")

    def allow_action(self, movement_pose, now):
        if movement_pose:
            self.last_movement = now
            return False
        return now - self.last_movement >= .2


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
