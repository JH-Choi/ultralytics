# Reference 
# https://docs.ultralytics.com/datasets/detect/visdrone/#citations-and-acknowledgments


from ultralytics.models.yolo.model import YOLO

data = 'VisDrone.yaml' # Okutama.yaml | Archangel.yaml
name = 'train_VisDrone' # experiment name
epochs = 100
freeze = 10
batch = 8
img=640

model = YOLO('./yolov8n.pt')  # load a pretrained model (recommended for training)

results = model.train(data=data, epochs=epochs, name=name, freeze=freeze, batch=batch, imgsz=img)  
results = model.val()  # evaluate model performance on the validation set