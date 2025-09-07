from ultralytics import YOLO
import cv2
import random

model = YOLO('weights/pill_detector.pt')  # updated model path

colors = {cls_id: (random.randint(0,255), random.randint(0,255), random.randint(0,255)) 
          for cls_id in model.names.keys()}

cap = cv2.VideoCapture(0)

count_history = []
buffer_size = 5  # smoothing buffer size

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.3)[0]
    
    count = 0
    for box in results.boxes:
        conf = box.conf[0].item()
        if conf < 0.3:
            continue
        count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0].item())
        label = model.names[cls]
        color = colors.get(cls, (0, 255, 0))

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
        cv2.rectangle(frame, (x1,y1-18), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    
    count_history.append(count)
    if len(count_history) > buffer_size:
        count_history.pop(0)
    smooth_count = int(round(sum(count_history) / len(count_history)))

    cv2.putText(frame, f'Count: {smooth_count}', (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),2)
    cv2.imshow('Live Pill Counting', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
