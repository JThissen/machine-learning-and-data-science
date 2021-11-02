# Recurrent neural network 1

Run the project using docker:
```bash
git clone git@github.com:JThissen/machine_learning.git
cd machine_learning/recurrent_neural_network_1
docker build -t recurrent_neural_network_1 -f build.Dockerfile .
docker run recurrent_neural_network_1
```

Or, if you'd rather run it locally *(make sure to uncomment `# torch==1.10.0+cu102` in `requirements.txt`)*:
```bash
git clone git@github.com:JThissen/machine_learning.git
cd machine_learning/recurrent_neural_network_1
pip install -r requirements.txt
python main.py
```
