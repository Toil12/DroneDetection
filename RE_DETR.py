from ultralytics import RTDETR


# Load a COCO-pretrained RT-DETR-l model


model = RTDETR("trained_models/rtdetr-l.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="../yolo_data_files/coco8.yaml", epochs=100, imgsz=640)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model("coco_test.jpg")