import os
import sys
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


def get_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = int(i * (180 / num_colors))
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        color = tuple(int(c) for c in bgr_color[0][0])
        colors.append(color)
    return colors


def get_emotion_label(emotion_model, emotion_results):
    names = emotion_model.names
    if not emotion_results:
        return "Unknown"

    result = emotion_results[0]

    if getattr(result, "probs", None) is not None and result.probs.top1 is not None:
        class_id = int(result.probs.top1)
        return names[class_id]

    if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
        class_id = int(result.boxes.cls[0].item())
        return names[class_id]

    return "Unknown"


def track_objects(frame, detections, colors, tracker):
    tracks = tracker.update_tracks(detections, frame=frame)
    person_count = 0
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        emotion_label = track.det_class

        color = colors[person_count % len(colors)]
        person_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"ID: {track_id} - {emotion_label}"
        font_scale = 0.6
        font_thickness = 1
        (label_width, label_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        cv2.rectangle(frame, (x1, y1), (x1 + label_width, y1 + label_height), color, -1)
        cv2.putText(
            frame,
            label_text,
            (x1, y1 + label_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

    return frame


class YOLOObjectDetectionGUI:
    def __init__(self):
        if getattr(sys, "frozen", False):
            self.base_path = sys._MEIPASS
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        seg_model_path = os.path.join(self.base_path, "yolo11n-seg.pt")
        try:
            self.seg_model = YOLO(seg_model_path)
        except Exception as e:
            print(f"实例分割模型加载失败: {e}")
            sys.exit(1)

        emotion_model_path = os.path.join(self.base_path, "trained_model.pt")
        try:
            self.emotion_model = YOLO(emotion_model_path)
        except Exception as e:
            print(f"情绪识别模型加载失败: {e}")
            sys.exit(1)

        self.root = tk.Tk()
        self.root.title("YOLO Object Detection")
        self.root.geometry("320x320")

        self.outer_frame = tk.Frame(self.root)
        self.outer_frame.pack(expand=True)

        self.button_frame = tk.Frame(self.outer_frame)
        self.button_frame.pack()

        self.create_buttons()

    def resize_image(self, image, target_width=640):
        height, width = image.shape[:2]
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        dim = (target_width, target_height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def process_frame(self, frame):
        seg_results = self.seg_model(frame)
        detections = []

        for r in seg_results:
            if r.masks is None:
                continue

            for j, _mask in enumerate(r.masks.xy):
                class_id = int(r.boxes.cls[j].item())
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, r.boxes.xyxy[j].cpu().numpy())
                person_image = frame[y1:y2, x1:x2]

                if person_image.size <= 0:
                    continue

                person_image = cv2.resize(person_image, (640, 640))
                emotion_results = self.emotion_model(person_image)
                emotion_label = get_emotion_label(self.emotion_model, emotion_results)
                detections.append(([x1, y1, x2 - x1, y2 - y1], 1.0, emotion_label))

        return detections

    def setup_tracking(self):
        tracker = DeepSort(max_age=5, n_init=2)
        colors = get_colors(10)
        return tracker, colors

    def show_frame(self, frame):
        resized_frame = self.resize_image(frame)
        cv2.imshow("YOLO Object Detection", resized_frame)
        return resized_frame

    def process_video(self, cap):
        all_results = []
        all_id_emotions = []
        try:
            tracker, colors = self.setup_tracking()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.process_frame(frame)
                tracks = tracker.update_tracks(detections, frame=frame)
                frame_id_emotions = []
                for track in tracks:
                    if track.is_confirmed():
                        track_id = track.track_id
                        emotion_label = track.det_class
                        frame_id_emotions.append((track_id, emotion_label))

                all_id_emotions.append(frame_id_emotions)

                processed_frame = track_objects(frame, detections, colors, tracker)
                resized_frame = self.show_frame(processed_frame)
                all_results.append(resized_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception as e:
            print(f"处理视频时出错: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return all_results, all_id_emotions

    def process_image(self, file_path):
        try:
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("无法读取图片文件")

            tracker, colors = self.setup_tracking()
            detections = self.process_frame(image)
            tracks = tracker.update_tracks(detections, frame=image)
            id_emotions = []
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    emotion_label = track.det_class
                    id_emotions.append((track_id, emotion_label))

            processed_image = track_objects(image, detections, colors, tracker)
            self.show_frame(processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return [processed_image], [id_emotions]
        except Exception as e:
            print(f"处理图片时出错: {e}")
            return [], []

    def detect_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            processed_image, id_emotions = self.process_image(file_path)
            print(id_emotions)

    def detect_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            cap = cv2.VideoCapture(file_path)
            all_results, all_id_emotions = self.process_video(cap)
            print(all_id_emotions)

    def detect_camera(self):
        cap = cv2.VideoCapture(0)
        all_results, all_id_emotions = self.process_video(cap)
        print(all_id_emotions)

    def create_buttons(self):
        image_button = tk.Button(self.button_frame, text="Select Image", command=self.detect_image, width=20, height=2)
        image_button.pack(pady=10)

        video_button = tk.Button(self.button_frame, text="Select Video", command=self.detect_video, width=20, height=2)
        video_button.pack(pady=10)

        camera_button = tk.Button(self.button_frame, text="Start Camera", command=self.detect_camera, width=20, height=2)
        camera_button.pack(pady=10)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = YOLOObjectDetectionGUI()
    app.run()
