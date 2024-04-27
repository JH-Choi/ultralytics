import ultralytics
#from ultralytics import YOLO
from ultralytics.models.yolo.model import YOLO

data = 'Okutama.yaml'
name = 'train_D1_Morn_val_D1_Morn'
epochs = 20
freeze = 10
batch = 16
img=1280

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from scratch
model = YOLO('./ckpts/yolov8n.pt')  # load a pretrained model (recommended for training)

# model = YOLO('yolov8s.yaml')  # build a new model from scratch
# model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

results = model.train(data=data, epochs=epochs, name=name, freeze=freeze, batch=batch, imgsz=img)  
results = model.val()  # evaluate model performance on the validation set


# results = model.val(data='Archangel.yaml')  # evaluate model performance on the validation set
