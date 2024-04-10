import ultralytics
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from scratch
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Load a model
# model = YOLO('yolov8s.yaml')  # build a new model from scratch
# model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)



results = model.train(data='Archangel.yaml', epochs=20)
results = model.val()  # evaluate model performance on the validation set


# results = model.val(data='Archangel.yaml')  # evaluate model performance on the validation set
