# Machine learning projects
Personal collection of my machine learning projects based on PyTorch. This is a WIP and more projects will be added over time. 

To run a project, such as [multi_class_image_classification](https://github.com/JThissen/machine_learning/tree/master/multi_class_image_classification), using docker:
```bash
git clone git@github.com:JThissen/machine_learning.git
cd machine_learning/multi_class_image_classification
docker build -t multi_class_image_classification -f build.Dockerfile .
docker run multi_class_image_classification
```

Or, if you'd rather run it locally *(make sure to uncomment `# torch==1.10.0+cu102` and `# torchvision==0.11.1+cu102` in `requirements.txt`)*:
```bash
git clone git@github.com:JThissen/machine_learning.git
cd machine_learning/multi_class_image_classification
pip install -r requirements.txt
python main.py
```

## Projects
This repository currently contains the following subdirectories, or projects:
- [multi_class_image_classification](https://github.com/JThissen/machine_learning/tree/master/multi_class_image_classification)