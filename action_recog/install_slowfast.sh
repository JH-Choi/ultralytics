export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

pip install -U iopath
pip install simplejson

git clone https://github.com/facebookresearch/slowfast
cd SlowFast
python setup.py build develop

# pip install pytorchvideo
cd ../thirdparty
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip isntall -e .

git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo


# from slowfast.models import build_model