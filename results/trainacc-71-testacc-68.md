## Network Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 64, 64]             160
         MaxPool2d-2           [-1, 16, 32, 32]               0
              ReLU-3           [-1, 16, 32, 32]               0
            Conv2d-4           [-1, 32, 32, 32]           4,640
         MaxPool2d-5           [-1, 32, 16, 16]               0
              ReLU-6           [-1, 32, 16, 16]               0
            Conv2d-7           [-1, 64, 16, 16]          18,496
       BatchNorm2d-8           [-1, 64, 16, 16]             128
         MaxPool2d-9             [-1, 64, 8, 8]               0
             ReLU-10             [-1, 64, 8, 8]               0
           Conv2d-11            [-1, 128, 8, 8]          73,856
      BatchNorm2d-12            [-1, 128, 8, 8]             256
        MaxPool2d-13            [-1, 128, 4, 4]               0
             ReLU-14            [-1, 128, 4, 4]               0
          Dropout-15            [-1, 128, 4, 4]               0
           Linear-16                 [-1, 1024]       2,098,176
          Dropout-17                 [-1, 1024]               0
           Linear-18                  [-1, 512]         524,800
          Dropout-19                  [-1, 512]               0
           Linear-20                    [-1, 2]           1,026
================================================================
Total params: 2,721,538
Trainable params: 0
Non-trainable params: 2,721,538
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 1.63
Params size (MB): 10.38
Estimated Total Size (MB): 12.03
----------------------------------------------------------------
```

### Hyperparameters
- dropout on conv: p=0.2
- dropout after fc: p=0.5
- learning rate: 0.001
- optimizer: adam
- weight decay: 1e-5
- batch size: 4096
- epochs: 100

### Train

**Accuracy:** 0.70981014226937

**Recall:** 0.70981014226937

**Precision:** 0.7075223450232364

**F1 Score:** 0.7081425675707399

|              | precision |  recall | f1-score | support |
|-------------|-----------|---------|----------|---------|
| No Findings |      0.74  |    0.78     | 0.76  |   47505|
|   Pneumonia |     0.66    |  0.61      |0.64   |  33187|
| | | | | |
|    accuracy  |           |           |   0.71   |  80692|
|   macro avg   |    0.70    |  0.70    |  0.70   |  80692|
| weighted avg    |   0.71    |  0.71  |    0.71  |   80692|

### Validation

**Accuracy:** 0.6788223486115758

**Recall:** 0.6788223486115758

**Precision:** 0.6715558529201899

**F1 Score:** 0.6670683146924375

|              | precision |  recall | f1-score | support |
|-------------|-----------|---------|----------|---------|
| No Findings |      0.63    |  0.46    |  0.53    |  4746|
|   Pneumonia |      0.70    |  0.82    |  0.76    |  7210|
| | | | | |
|    accuracy  |           |           |    0.68   |  11956|
|   macro avg   |    0.66    |  0.64    |  0.64    | 11956|
| weighted avg    |   0.67   |   0.68  |    0.67   |  11956|


### Test

**Accuracy:** 0.6847344207444583

**Recall:** 0.6847344207444583

**Precision:** 0.676000451937918

**F1 Score:** 0.6736373790393815

|              | precision |  recall | f1-score | support |
|-------------|-----------|---------|----------|---------|
| No Findings |     0.62   |   0.47    |  0.53     | 4597|
|   Pneumonia |       0.71  |    0.82   |   0.76   |   7358|
| | | | | |
|    accuracy  |           |           |    0.68   |  11955|
|   macro avg   |    0.67    |  0.64   |   0.65    | 11955|
| weighted avg    |  0.68    |  0.68  |    0.67    | 11955|
