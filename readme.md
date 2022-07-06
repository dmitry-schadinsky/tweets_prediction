 # Deep Learning model to predict geolocation of tweets

 ## General information

 ### Task
 The goal is to create and train a deep learning model which predicts coordinates (latitude, longitude) of individual tweets.

### Data
 The data consists of csv files where each line contains directly text itself and information about the location of the area (bottom left and top right coordinates of a "rectangular") where the tweet was made. (Data contains tweets from South America)

 ### Model
 Here a generative approach is considered. The proposed model takes UTF-8 encoding text as input. The model architecture consists of  a character embedding layer followed by the series of CNN, BN and Pooling layers. As a result N coefficients are obtained which represent the coefficients of decomposition by basis of N von Mises-Fisher PDFs. The parameters of the basis functions are trained as well as layers' parameters.

## Training procedure
### General
First of all the train and test datasets are prepared. 95% of all data is considered as the train part, while the rest is the test. The raw information is processed and saved in corresponding files for further training. The average values of the input data are calculated so normalized data are considered. Also the cleaning for data is implied. Only tweets which "rectangular" area's "diagonal" is less than 500 km are taken into account. Otherwise such data only confuses our model during training.

After this the training loop starts. Each epoch the statistic is provided, which includes average test error values, percentiles and histograms for test errors. 

### Source code files description

- [train.py](./train.py) - the base training script
- [dataset.py](./dataset.py) - source code regarding datasets
- [model.py](./model.py) - model class and support functions
- [helper.py](./helper.py) - the other utility functions
- [hparam.py](./hparam.py) - the file with hyper parameters of the training process and other settings
- [inference.py](./inference.py) - script for processing data through the saved model

## Results
Here, you can see the results of test data errors.

![This is an image](./results/hist.png)

- Average error distance = 703.76 km
- Quantile 0.9 = 1648.51 km
- Quantile 0.75 = 840.65 km
- Quantile 0.5 = 356.89 km
