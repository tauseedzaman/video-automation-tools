import cv2
import numpy as np
from moviepy import VideoFileClip
import argparse
import os
import sys


class SmartVerticalTracker:

    def __init__(self, width, height):
        self.target_w = width
        self.target_h = height

        self.prev_gray = None
        self.locked_cx = None
        self.locked_cy = None
        self.prev_cx = None
        self.prev_cy = None

        self.dead_zone = 15
        self.motion_threshold = 100
        self.stickiness_frames = 120

        self.no_motion_counter = 0
        self.transition_progress = 1.0
        self.transition_from_cx = None
        self.transition_from_cy = None

        # Kalman filter
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]
        ], np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.005
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 8


    def detect_cursor(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        if self.prev_gray is None:
            self.prev_gray = gray
            return None, None, 0

        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.prev_gray = gray.copy()

        if not contours:
            return None, None, 0

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area < self.motion_threshold:
            return None, None, area

        x, y, wc, hc = cv2.boundingRect(c)
        cx = x + wc // 2
        cy = y + hc // 2

        return cx, cy, area


    def smooth_transition(self, target_cx, target_cy):
        if self.transition_progress >= 1.0:
            return target_cx, target_cy

        if self.transition_from_cx is None or self.transition_from_cy is None:
            self.transition_progress = 1.0
            return target_cx, target_cy

        self.transition_progress += 0.08
        t = min(1.0, self.transition_progress)
        t = 1 - (1 - t) ** 2   # ease-out

        cx = int(self.transition_from_cx + (target_cx - self.transition_from_cx) * t)
        cy = int(self.transition_from_cy + (target_cy - self.transition_from_cy) * t)
        return cx, cy


    def decide_position(self, new_cx, new_cy, area):
        if new_cx is None:
            if self.locked_cx is not None:
                self.no_motion_counter += 1
                return self.locked_cx, self.locked_cy
            return self.target_w // 2, self.target_h // 2

        strong = area > self.motion_threshold * 1.6

        if strong:
            if self.locked_cx is not None and self.locked_cy is not None:
                self.transition_from_cx = self.locked_cx
                self.transition_from_cy = self.locked_cy
                self.transition_progress = 0.0

            self.locked_cx = new_cx
            self.locked_cy = new_cy
            self.no_motion_counter = 0

            if self.transition_from_cx is not None and self.transition_progress < 1.0:
                return self.smooth_transition(new_cx, new_cy)
            return new_cx, new_cy

        if self.locked_cx is None:
            self.locked_cx = new_cx
            self.locked_cy = new_cy
            self.no_motion_counter = 0
            return new_cx, new_cy

        if self.transition_progress < 1.0:
            return self.smooth_transition(self.locked_cx, self.locked_cy)

        return self.locked_cx, self.locked_cy


    def apply_deadzone(self, cx, cy):
        if self.prev_cx is None:
            return cx, cy
        if abs(cx - self.prev_cx) < self.dead_zone:
            cx = self.prev_cx
        if abs(cy - self.prev_cy) < self.dead_zone:
            cy = self.prev_cy
        return cx, cy

    def crop_vertical(self, frame, cx, cy):
        h, w, _ = frame.shape

        target_ratio = self.target_w / self.target_h

        # compute crop size based on ratio
        crop_h = h
        crop_w = int(crop_h * target_ratio)

        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / target_ratio)

        half_w = crop_w // 2
        half_h = crop_h // 2

        cy -= int(crop_h * 0.14)

        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)

        x2 = x1 + crop_w
        y2 = y1 + crop_h

        if x2 > w:
            x1 = w - crop_w
        if y2 > h:
            y1 = h - crop_h

        crop = frame[y1:y2, x1:x2]

        # resize AFTER cropping
        crop = cv2.resize(crop, (self.target_w, self.target_h), interpolation=cv2.INTER_LANCZOS4)

        return crop

    def process_frame(self, frame):
        h, w, _ = frame.shape

        new_cx, new_cy, area = self.detect_cursor(frame)

        cx, cy = self.decide_position(new_cx, new_cy, area)

        # Initialize Kalman on first real detection
        if self.prev_cx is None and new_cx is not None:
            self.kalman.statePre = np.array([[float(cx)], [float(cy)], [0.0], [0.0]], np.float32)

        # Apply smoothing only during significant motion
        if new_cx is not None and area > self.motion_threshold:
            cx, cy = self.smooth(cx, cy)

        # camera easing
        alpha = 0.12
        if self.prev_cx is not None:
            cx = int(self.prev_cx * (1 - alpha) + cx * alpha)
            cy = int(self.prev_cy * (1 - alpha) + cy * alpha)

        crop = self.crop_vertical(frame, cx, cy)

        self.prev_cx = cx
        self.prev_cy = cy

        return crop


    def smooth(self, cx, cy):
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(measurement)
        pred = self.kalman.predict()
        return int(pred[0, 0]), int(pred[1, 0])   # ← FIXED HERE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("Input video not found")
        sys.exit(1)

    output = args.output or os.path.splitext(args.input)[0] + "_vertical.mp4"

    print("Loading video...")
    clip = VideoFileClip(args.input)

    tracker = SmartVerticalTracker(args.width, args.height)

    print("Processing frames...")
    vertical = clip.image_transform(tracker.process_frame)

    print("Rendering video...")
    vertical.write_videofile(
        output,
        codec="libx264",
        audio_codec="aac",
        threads=5,
        preset="slow",
        ffmpeg_params=["-crf", "21"]
    )

    print("Done:", output)


if __name__ == "__main__":
    main()