from pathlib import Path
import cv2
import numpy as np


def detect_face_tracks(path: Path, start: float, end: float) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float, float]]]:
    """Return group centers and the most prominent face track in sampled frames."""
    cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return [], []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return [], []
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        return [], []

    group_points: list[tuple[float, float, float]] = []
    focus_points: list[tuple[float, float, float, float]] = []
    sample_count = min(40, max(8, int(np.ceil((end - start) / 1.5))))
    sample_times = np.linspace(start, end, sample_count, endpoint=False)
    for second in sample_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        scale = min(1.0, 640 / frame.shape[1])
        sample = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(32, 32))
        if len(faces):
            # Discard small detections that are more likely background noise.
            faces = [box for box in faces if (box[2] * box[3]) / (sample.shape[0] * sample.shape[1]) >= .008]
            if not faces:
                continue
            prominent = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[:4]
            weights = np.array([np.sqrt(box[2] * box[3]) for box in prominent], dtype=np.float32)
            centers_x = np.array([box[0] + box[2] / 2 for box in prominent], dtype=np.float32)
            centers_y = np.array([box[1] + box[3] / 2 for box in prominent], dtype=np.float32)
            center_x = float(np.average(centers_x, weights=weights)) / sample.shape[1]
            center_y = float(np.average(centers_y, weights=weights)) / sample.shape[0]
            clip_time = float(second - start)
            group_points.append((clip_time, center_x, center_y))
            x, y, face_width, face_height = prominent[0]
            focus_points.append((clip_time, (x + face_width / 2) / sample.shape[1], (y + face_height / 2) / sample.shape[0], face_height / sample.shape[0]))
    capture.release()
    # A one-frame Haar false positive must never trigger a full-screen crop.
    # Require a nearby detection that stays in roughly the same place and size.
    stable_focus = []
    for point in focus_points:
        time, x, y, face_height = point
        for other_time, other_x, other_y, other_height in focus_points:
            if other_time == time or abs(other_time - time) > 3.1:
                continue
            distance = ((other_x - x) ** 2 + (other_y - y) ** 2) ** .5
            size_ratio = min(face_height, other_height) / max(face_height, other_height)
            if distance <= .18 and size_ratio >= .5:
                stable_focus.append(point)
                break
    stable_times = {point[0] for point in stable_focus}
    stable_groups = [point for point in group_points if point[0] in stable_times]
    return stable_groups, stable_focus


def detect_face_track(path: Path, start: float, end: float) -> list[tuple[float, float, float]]:
    """Compatibility helper returning the detected group-center track."""
    group_points, _focus_points = detect_face_tracks(path, start, end)
    return group_points


def analyze_video(path: Path, start: float, end: float) -> dict:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    duration = max(0, end - start)
    step = max(1, int(fps * 1.5))
    frames, motion, cuts = 0, [], 0
    previous = None
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    stop_frame = int(end * fps)
    frame_no = int(start * fps)
    while frame_no < stop_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        if previous is not None:
            difference = float(np.mean(cv2.absdiff(gray, previous))) / 255
            motion.append(difference)
            cuts += int(difference > .22)
        previous = gray
        frames += 1
        frame_no += step
    capture.release()
    if frames == 0:
        raise RuntimeError("The video frames could not be analyzed.")
    avg_motion = float(np.mean(motion)) if motion else 0
    score = max(0, min(10, avg_motion * 30 + min(cuts / max(1, duration / 8), 3) * 1.8))
    return {"score": round(score, 1), "motion": round(avg_motion, 3), "scene_changes": cuts, "sampled_frames": frames}
