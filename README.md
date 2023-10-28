<<<<<<< HEAD
#### **1、Environment and Installation**

We are using pytorch gpu environment, we default that you have installed Anaconda. Once you have installed Anaconda, open the Anaconda Prompt and enter the following statements to install pytorch gpu, including installing Python>=3.7.0 environment and PyTorch>=1.7.

```python
conda create -n pytorch_gpu python=3.7
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install -r requirements.txt  # install
```

#### **2、Code Resource Description**

You will need the following instructions to copy our code resources:

```python
git clone https://github.com/wuchongxin/yolov5_LEAS.git  # clone our code source
```

The model and utils folder are supporting files for training, `train.py`file is the file we use for training, `val.py` is to verify whether overfitting, and used to adjust the training parameters, etc., in `test.py` file we will import the trained model to test the effectiveness using test set (in detect folder). `requirements.txt` is the packages we need for training.

The data folder contains your dataset and a yaml file containing the paths to the data, as shown in the following example. The labels is in txt file format, and the names in the images and labels must be the same.

```python
# Schematic structure of the downloaded resources
yolov5_LEAS
  	- model
    	- yolov5n.yaml
        - yolov5n_LEAS.yaml
  	- utils
        - data
            - dataset.yaml
            - dataset
                  - train
                        - images
                        - labels
                  - val
                        - images
                        - labels
                  - detect
                    	- images
                        - labels
        - train.py
        - val.py
        - detect.py
        - requirements.txt
        - yolov5n.pt
```

#### **3、Model Training and Detecting**

When we want to train this model with our own dataset, you need to set up the path to your dataset, the number of categories to train and the corresponding category names in the dataset.yaml document in the data folder.

The `train` is the training set, and the `val` is the test set during the training process, which is designed to allow you to see the results of the training as you go along and determine the learning state in time. `test` is the test set for evaluating the results of the model after training it. Only train can be trained, val is not required, and the scale can be set very small. `test` is also not necessary for model training, but generally need to set aside some for testing, usually the recommended ratio is 8:1:1.

The `val` is short for `validation`. Both `training dataset` and `validation dataset` work at the time of training. And since the `validation` dataset does not intersect with `training`, this part of the data does not contribute to the final trained model. The main purpose of `validation` is to verify whether the model is overfitted or not, and to adjust the training parameters.

For example, the `loss` of `train` and `validation` are decreasing during `0-10,000` iterations, but the `train loss` is decreasing during `10,000-20,000`, and the `loss` of `validation` is increasing instead of decreasing. Then it proves that if we continue to train, the model only fits the part of `training dataset` exceptionally well, but the generalization ability is very poor. So instead of picking `20,000` times the result, we should pick `10,000` times the result. The name of this process is `Early Stop`, and `validation` data is essential in this process.

**1. **Let's start by opening Anaconda Prompt and typing the following statement to activate the pytorch gpu environment and get to the path where the code is located.

```python
conda activate pytorch_gpu
cd ./yolov5_LEAS
```

**2. **Run the `train.py` file and execute the following commands:

```python
python train.py --weights yolov5n.pt
				--cfg models/yolov5_LEAS.yaml   
    			--data data/dataset.yaml       # your dataset path
        		--epochs 300				   # number of training iterations
```

The --weights is the initial weights path, we train the network is generally to use the official website has been trained weights followed by training, so that the results of the training out of the better. The --cfg defines which model to use, yolov5 gives five models, `yolov5n.yaml` is the model with the smallest number of parameters, is also the fastest model to train, but the accuracy may not be as high as other models. Here, it needs to be changed to our model's corresponding `yolov5_LEAS.yaml` to accommodate our model's modified model configuration. 

**3. **Run the `val.py` file and execute the following command:

```python
python val.py --data data/dataset.yaml
			  --weights model.pt
```

Where `model.pt` is what you get after training your dataset.

**4. **Run the `detect.py` file and execute the following commands:

```python
python detect.py --data data/dataset.yaml
				 --weights model.pt
```

=======
#### **1、Environment and Installation**

We are using pytorch gpu environment, we default that you have installed Anaconda. Once you have installed Anaconda, open the Anaconda Prompt and enter the following statements to install pytorch gpu, including installing Python>=3.7.0 environment and PyTorch>=1.7.

```python
conda create -n pytorch_gpu python=3.7
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install -r requirements.txt  # install
```

#### **2、Code Resource Description**

You will need the following instructions to copy our code resources:

```python
git clone https://github.com/wuchongxin/yolov5_LEAS.git  # clone our code source
```

The model and utils folder are supporting files for training, `train.py`file is the file we use for training, `val.py` is to verify whether overfitting, and used to adjust the training parameters, etc., in `test.py` file we will import the trained model to test the effectiveness using test set (in detect folder). `requirements.txt` is the packages we need for training.

The data folder contains your dataset and a yaml file containing the paths to the data, as shown in the following example. The labels is in txt file format, and the names in the images and labels must be the same.

```python
# Schematic structure of the downloaded resources
yolov5_LEAS
  	- model
    	- yolov5n.yaml
        - yolov5n_LEAS.yaml
  	- utils
        - data
            - dataset.yaml
            - dataset
                  - train
                        - images
                        - labels
                  - val
                        - images
                        - labels
                  - detect
                    	- images
                        - labels
        - train.py
        - val.py
        - detect.py
        - requirements.txt
        - yolov5n.pt
```

#### **3、Model Training and Detecting**

When we want to train this model with our own dataset, you need to set up the path to your dataset, the number of categories to train and the corresponding category names in the dataset.yaml document in the data folder.

The `train` is the training set, and the `val` is the test set during the training process, which is designed to allow you to see the results of the training as you go along and determine the learning state in time. `test` is the test set for evaluating the results of the model after training it. Only train can be trained, val is not required, and the scale can be set very small. `test` is also not necessary for model training, but generally need to set aside some for testing, usually the recommended ratio is 8:1:1.

The `val` is short for `validation`. Both `training dataset` and `validation dataset` work at the time of training. And since the `validation` dataset does not intersect with `training`, this part of the data does not contribute to the final trained model. The main purpose of `validation` is to verify whether the model is overfitted or not, and to adjust the training parameters.

For example, the `loss` of `train` and `validation` are decreasing during `0-10,000` iterations, but the `train loss` is decreasing during `10,000-20,000`, and the `loss` of `validation` is increasing instead of decreasing. Then it proves that if we continue to train, the model only fits the part of `training dataset` exceptionally well, but the generalization ability is very poor. So instead of picking `20,000` times the result, we should pick `10,000` times the result. The name of this process is `Early Stop`, and `validation` data is essential in this process.

**1. **Let's start by opening Anaconda Prompt and typing the following statement to activate the pytorch gpu environment and get to the path where the code is located.

```python
conda activate pytorch_gpu
cd ./yolov5_LEAS
```

**2. **Run the `train.py` file and execute the following commands:

```python
python train.py --weights yolov5n.pt
				--cfg models/yolov5_LEAS.yaml   
    			--data data/dataset.yaml       # your dataset path
        		--epochs 300				   # number of training iterations
```

The --weights is the initial weights path, we train the network is generally to use the official website has been trained weights followed by training, so that the results of the training out of the better. The --cfg defines which model to use, yolov5 gives five models, `yolov5n.yaml` is the model with the smallest number of parameters, is also the fastest model to train, but the accuracy may not be as high as other models. Here, it needs to be changed to our model's corresponding `yolov5_LEAS.yaml` to accommodate our model's modified model configuration. 

**3. **Run the `val.py` file and execute the following command:

```python
python val.py --data data/dataset.yaml
			  --weights model.pt
```

Where `model.pt` is what you get after training your dataset.

**4. **Run the `detect.py` file and execute the following commands:

```python
python detect.py --data data/dataset.yaml
				 --weights model.pt
```

>>>>>>> b149f045b528ca0989146b97729caf7f3b0a7ffb
 And finally, you can find the results of the corresponding experiment in the runs folder.