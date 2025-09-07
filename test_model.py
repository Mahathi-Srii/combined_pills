from ultralytics import YOLO

def main():
    # Load your custom trained YOLOv9c model weights
    model = YOLO('weights/pill_detector.pt')  # Update filename/path if needed

    # Path to an image for testing inference
    test_image_path = 'test/images/example.jpg'  # Update with actual image path

    # Run inference
    results = model(test_image_path)

    # Display detection results (shows image with predicted bounding boxes)
    results.show()

    # Save inference results to 'yolo_outputs' folder
    results.save(save_dir='yolo_outputs')

if __name__ == '__main__':
    main()
