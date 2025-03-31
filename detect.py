import cv2
import matplotlib.pyplot as plt


def detect_target_objects(yolo_model, image_path, target_object, show_image=True):
    """
    Detects objects in an image using YOLOv8 and filters results for the target object.
    Draws bounding boxes on the matching objects (with confidence ≥ threshold).
    Returns a list of detections.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return None

    # Run YOLO detection
    results = yolo_model(image)
    class_names = yolo_model.names  # YOLO model's class names
    detected_bboxes = []

    # Process detections
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = box.conf.item()
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            detected_class = class_names[cls_id]
            
            # Filter by target object name & confidence threshold
            if target_object.lower() in detected_class.lower() and confidence >= 0.5:
                detected_bboxes.append({
                    "class": detected_class,
                    "confidence": confidence,
                    "bbox": bbox.tolist()
                })
                # Draw bounding box and label
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                cv2.putText(image, f"{detected_class}: {confidence*100:.2f}%", 
                            (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
    # Display the image using matplotlib (suitable for both Jupyter and .py files)
    if show_image:
        plt.imshow(image[:, :, ::-1])  # Convert BGR to RGB
        plt.axis("off")
        plt.show()
    
    return detected_bboxes