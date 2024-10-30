# Software
We provide the software in the 'software' directory. Please download the finetuned models  from the [Zenodo repository](https://doi.org/10.5281/zenodo.13622929) or [Hugging Face](https://huggingface.co/Orange066/Unmixing_Model) and the examples from the [Zenodo repository](https://doi.org/10.5281/zenodo.13622692)  or [Hugging Face](https://huggingface.co/datasets/Orange066/Unmixing_ExampleData) of EndmemberNet. Then, you can 'tar -xzvf' the file and put the folder in the root directory of EndmemberNet. 

```
camera/
    exampledata/
        Unmixing/
    checkpoints/
        detection/
        segmentation/
        
example/ 
    exampledata/
        Unmixing/
    checkpoints/
        detection/
        segmentation/
```

In the 'software/camera' folder, you can use a Princeton camera to capture fluorescence images and perform real-time unmixing, provided you **install the LightField software and set the 'LIGHTFIELD_ROOT' environment variable**. If the 'LIGHTFIELD_ROOT' path is not set, you can run the 'main.exe' file in the 'software/example' folder and click 'Use Example' and 'Extract Spectrum' to view the example unmixing results from our paper.
