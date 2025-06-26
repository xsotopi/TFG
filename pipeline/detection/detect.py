import cv2
import matplotlib.pyplot as plt


def detect_target_objects(yolo_model, image_path, target_object, show_image=True):
    """
    Detects objects in an image using YOLOv8 and filters results for the target object.
    Draws bounding boxes on the matching objects (with confidence ≥ threshold).
    Returns a list of detections.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return None

    results = yolo_model(image)
    class_names = yolo_model.names
    detected_bboxes = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = box.conf.item()
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            detected_class = class_names[cls_id]
            
            if target_object.lower() in detected_class.lower() and confidence >= 0.5:
                detected_bboxes.append({
                    "class": detected_class,
                    "confidence": confidence,
                    "bbox": bbox.tolist()
                })
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                cv2.putText(image, f"{detected_class}: {confidence*100:.2f}%", 
                            (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
    if show_image:
        plt.imshow(image[:, :, ::-1])
        plt.axis("off")
        plt.show()
    
    return detected_bboxes


def detect_target_objects_realtime(yolo_model, frame, target_object, show_image=False):
    """
    Detects objects in a real-time frame using YOLOv8, draws bounding boxes
    on the matching objects (modifies the frame in-place), and returns the list of detections.
    """
    if frame is None:
        print("Error: Received None frame for detection!")
        return []

    class_names = yolo_model.names
    detected_bboxes = []

    results = yolo_model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = box.conf.item()
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            detected_class = class_names[cls_id]

            if target_object.lower() in detected_class.lower() and confidence >= 0.5:
                detected_bboxes.append({
                    "class": detected_class,
                    "confidence": confidence,
                    "bbox": bbox.tolist()
                })
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detected_class}: {confidence * 100:.2f}%",
                            (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if show_image:
        cv2.imshow("Realtime Detection Debug", frame)
        cv2.waitKey(1)

    return detected_bboxes

