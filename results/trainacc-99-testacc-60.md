## Network Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 64, 64]             640
         MaxPool2d-2           [-1, 64, 32, 32]               0
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,856
         MaxPool2d-5          [-1, 128, 16, 16]               0
              ReLU-6          [-1, 128, 16, 16]               0
            Conv2d-7          [-1, 256, 16, 16]         295,168
         MaxPool2d-8            [-1, 256, 8, 8]               0
              ReLU-9            [-1, 256, 8, 8]               0
           Conv2d-10            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-11            [-1, 512, 4, 4]               0
             ReLU-12            [-1, 512, 4, 4]               0
          Dropout-13            [-1, 512, 4, 4]               0
           Linear-14                 [-1, 1024]       8,389,632
          Dropout-15                 [-1, 1024]               0
           Linear-16                  [-1, 256]         262,400
          Dropout-17                  [-1, 256]               0
           Linear-18                    [-1, 2]             514
================================================================
Total params: 10,202,370
Trainable params: 0
Non-trainable params: 10,202,370
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 5.71
Params size (MB): 38.92
Estimated Total Size (MB): 44.64
----------------------------------------------------------------
```

### Hyperparameters
- dropout on conv: p=0.2
- dropout after fc: p=0.5
- learning rate: 0.001
- optimizer: adam
- weight decay: 1e-5
- batch size: 1024
- epochs: 100

### Train

**Accuracy:** 0.9979180092202449

**Recall:** 0.9979180092202449

**Precision:** 0.9979191941352389

**F1 Score:** 0.997917638366897

|              | precision |  recall | f1-score | support |
|-------------|-----------|---------|----------|---------|
| No Findings |      1.00 |     1.00|      1.00|     47505|
|   Pneumonia |      1.00  |    1.00  |    1.00   |  33187|
| | | | | |
|    accuracy  |           |           |   1.00   |  80692|
|   macro avg   |    1.00  |    1.00    |  1.00   |  80692|
| weighted avg    |   1.00  |    1.00     | 1.00   |  80692|

### Validation

**Accuracy:** 0.6048845767815323

**Recall:** 0.6048845767815323

**Precision:** 0.6097521373626952

**F1 Score:** 0.6069196174366777

|              | precision |  recall | f1-score | support |
|-------------|-----------|---------|----------|---------|
| No Findings |      0.50  |    0.53  |    0.51   |   4679
|   Pneumonia |      0.68   |   0.65|      0.67     | 7277|
| | | | | |
|    accuracy  |           |           |    0.60  |   11956|
|   macro avg   |    0.59    |  0.59   |   0.59  |   11956|
| weighted avg    |   0.61   |   0.60   |   0.61   |  11956|


### Test

**Accuracy:** 0.6

**Recall:** 0.6

**Precision:** 0.6067482026587573

**F1 Score:** 0.6027176878121594

|              | precision |  recall | f1-score | support |
|-------------|-----------|---------|----------|---------|
| No Findings |      0.48   |   0.52 |     0.50  |    4610|
|   Pneumonia |      0.68     | 0.65    |  0.67  |    7345|
| | | | | |
|    accuracy  |           |           |    0.60  |   11955|
|   macro avg   |    0.58    |  0.59   |   0.58   |  11955|
| weighted avg    |  0.61   |   0.60    |  0.60  |   11955|
