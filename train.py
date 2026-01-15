import argparse
#from ultralytics import YOLO
from ultralytics.models.yolo.model import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Okutama.yaml', help='dataset.yaml')
parser.add_argument('--name', type=str, default='train_D1_Morn_val_D1_Morn', help='experiment name')
parser.add_argument('--model_name', type=str, default=None, help='model name')
args = parser.parse_args()

# data = 'Okutama.yaml' # Okutama.yaml | Archangel.yaml
# name = 'train_D1_Morn_val_D1_Morn' # experiment name
data = args.data # Okutama.yaml | Archangel.yaml
name = args.name # experiment name
epochs = 20
# epochs = 10
freeze = 10
batch = 16
# img=1280
img=640

# Load a model
if args.model_name == 'yolov8n':
    model = YOLO('yolov8n.yaml')  # build a new model from scratch
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
elif args.model_name == 'yolov8s':
    model = YOLO('yolov8s.yaml')  # build a new model from scratch
    model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
elif args.model_name == 'yolov8m':
    model = YOLO('yolov8m.yaml')  # build a new model from scratch
    model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)


results = model.train(data=data, epochs=epochs, name=name, freeze=freeze, batch=batch, imgsz=img)  

# results = model.val()

# evaluate model performance on the validation set
# results = model.val(data=data, split='val', save_json=True) 


# results = model.val(data='Archangel.yaml')  # evaluate model performance on the validation set
