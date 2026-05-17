from .image import resize_image


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


def process_frame(frame, seg_model, emotion_model):
    seg_results = seg_model(frame)
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

            person_image = resize_image(person_image, 640)
            emotion_results = emotion_model(person_image)
            emotion_label = get_emotion_label(emotion_model, emotion_results)
            detections.append(([x1, y1, x2 - x1, y2 - y1], 1.0, emotion_label))

    return detections
