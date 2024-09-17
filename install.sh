conda create -n yolov8 python=3.8
pip install -e . 


# Action recognition
pip install transformers

# Install PyTorchVideo
pip install cython
cd pytorchvideo
pip install -e .

pip uninstall av
# https://stackoverflow.com/questions/72604912/cant-show-image-with-opencv-when-importing-av
# av conflict with opencv
sudo apt-get install libavformat-dev libavdevice-dev
pip install av --no-binary av