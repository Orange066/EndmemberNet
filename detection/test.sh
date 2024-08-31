CUDA_VISIBLE_DEVICES=0 python detect.py --weights runs/train/unmixing/weights/best.pt --source data/unmixing/fluorescence_time_data/0006/ --save-txt --save-conf --save-crop   --conf-thres 0.05 --name unmixing_06

CUDA_VISIBLE_DEVICES=0 python detect.py --weights runs/train/unmixing/weights/best.pt --source data/unmixing/fluorescence_time_data/0016/ --save-txt --save-conf --save-crop   --conf-thres 0.05 --name unmixing_16

CUDA_VISIBLE_DEVICES=0 python detect.py --weights runs/train/unmixing/weights/best.pt --source data/unmixing/fluorescence_time_data/0019/ --save-txt --save-conf --save-crop   --conf-thres 0.05 --name unmixing_19

CUDA_VISIBLE_DEVICES=0 python detect.py --weights runs/train/unmixing/weights/best.pt --source data/unmixing/fluorescence_time_data/0020/ --save-txt --save-conf --save-crop   --conf-thres 0.05 --name unmixing_20

CUDA_VISIBLE_DEVICES=0 python detect.py --weights runs/train/unmixing/weights/best.pt --source data/unmixing/fluorescence_time_data/0021/ --save-txt --save-conf --save-crop   --conf-thres 0.05 --name unmixing_21

CUDA_VISIBLE_DEVICES=0 python detect.py --weights runs/train/unmixing/weights/best.pt --source data/unmixing/fluorescence_time_data/0025/ --save-txt --save-conf --save-crop   --conf-thres 0.05 --name unmixing_25

CUDA_VISIBLE_DEVICES=0 python detect.py --weights runs/train/unmixing/weights/best.pt --source data/unmixing/fluorescence_time_data/0026/ --save-txt --save-conf --save-crop   --conf-thres 0.05 --name unmixing_26

CUDA_VISIBLE_DEVICES=0 python detect.py --weights runs/train/unmixing/weights/best.pt --source data/unmixing/fluorescence_time_data/0027/ --save-txt --save-conf --save-crop   --conf-thres 0.05 --name unmixing_27

CUDA_VISIBLE_DEVICES=0 python detect.py --weights runs/train/unmixing/weights/best.pt --source data/unmixing/fluorescence_time_data/0028/ --save-txt --save-conf --save-crop   --conf-thres 0.05 --name unmixing_28
