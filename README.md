# dnn-ex3

## Question 1

### Training execution:

```python
    python Ex3_1.py
```

Comments:
1. Executing the VGG took very very very long time to train, so I have run with resnet50 instead due to lower memory footprint. 

### Train Execution result on various models:

1. 2 Conv - 2 FC Layers

    ![loss](Q1/Results/2Conv2FC/loss.png)

    ![accuarcy](Q1/Results/2Conv2FC/accuarcy.png)

    ![loss](Q1/Results/2Conv2FC/finegrainedloss.png)

    ![wieghts](Q1/Results/2Conv2FC/weights.png)

 1. 1 Conv 1 FC Layer
 
    ![loss](Q1/Results/1Conv1FC/loss.png)

    ![accuracy](Q1/Results/1Conv1FC/accuarcy.png)

    ![loss](Q1/Results/1Conv1FC/finegrainedloss.png)

    ![wieghts](Q1/Results/1Conv1FC/weights.png)

 1. 2 FC Layers

    ![loss](Q1/Results/2Fc/loss.png)

    ![accuarcy](Q1/Results/2Fc/accuarcy.png)

    ![loss](Q1/Results/2Fc/finegrainedloss.png)

    ![wieghts](Q1/Results/2Fc/weights.png) 

1. 1 FC Layers

    ![loss](Q1/Results/1Fc/loss.png)

    ![accuarcy](Q1/Results/1Fc/accuarcy.png)

    ![loss](Q1/Results/1Fc/finegrainedloss.png)

    ![wieghts](Q1/Results/1Fc/weights.png)

## Question 2

1. Train/test/valid datasets can be found on [here](https://drive.google.com/drive/folders/19PduWT-tEPssgwip1rgWiVApFRsl8Wkd?usp=sharing)
2. Training execution:
```python
    python Ex3_2_models.py
```
 it can take a while ...

5. Results:

 ![loss](Q2/Results/loss2.png)

 ![accuarcy](Q2/Results/accuracy.png)
 
 

