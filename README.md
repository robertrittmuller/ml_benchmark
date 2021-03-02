# Apple M1 Machine Learning Benchmark
Simple machine learning benchmark for Apple M1 and other platforms. 

## Installation / Setup
Installation can be performed on any platform that supports Tensorflow.

### Step One
Clone this repository:
```
git clone https://github.com/robertrittmuller/ml_benchmark
```

### Step Two
Install Apple-optimized Tensorflow from GitHub:
https://github.com/apple/tensorflow_macos

Make sure to follow the installation instructions that cover setting up a virtual environment! (Anaconda works also for non-M1 platforms)

## Example Results
### Dell 5820 Precision Workstation / Nvidia RTX Quadro 5000 16GB


    Architecture    Test Name	                                    Result
    x86_64	        Simple Neural Network Test	                    00:26.9
    x86_64	        RELU Activation Test                            00:37.4
    x86_64	        TANH Activation Test	                        00:37.2
    x86_64	        SIGMOID Activation Test	                        00:37.1
    x86_64	        Resnet50 Fine Tuning Test	                    00:19.4
    x86_64	        MobilenetV2 Fine Tuning Test                    00:16.3
    x86_64	        Inception V3 Inference Test (Batch size = 4)    00:19.1
    x86_64	        Resnet 50 Inference Test (Batch size = 4)	    00:19.5
    x86_64	        EfficientNet-B0 Inference Test (Batch size = 4) 00:16.8
    x86_64	        MobilenetV2 Inference Test (Batch size = 4)	    00:19.3
    x86_64	        Xception Inference Test (Batch size = 4)	    00:22.8

### AMD Ryzen 3900XT / Nvidia 2080 SUPER 8GB
    Architecture    Test Name	                                    Result
    AMD64	        Simple Neural Network Test	                    00:16.2
    AMD64	        RELU Activation Test	                        00:32.4
    AMD64	        TANH Activation Test	                        00:32.1
    AMD64	        SIGMOID Activation Test	                        00:31.8
    AMD64	        Resnet50 Fine Tuning Test	                    00:13.6
    AMD64	        MobilenetV2 Fine Tuning Test	                00:07.3
    AMD64	        Inception V3 Inference Test (Batch size = 4)    00:20.2
    AMD64	        Resnet 50 Inference Test (Batch size = 4)	    00:13.6
    AMD64	        EfficientNet-B0 Inference Test (Batch size = 4)	00:14.5
    AMD64	        MobilenetV2 Inference Test (Batch size = 4)	    00:08.5
    AMD64	        Xception Inference Test (Batch size = 4)	    00:16.6

### Apple M1 16GB Mac Mini
    Architecture    Test Name	                                    Result
    arm64	        Simple Neural Network Test	                    00:03.6
    arm64	        RELU Activation Test	                        00:50.2
    arm64	        TANH Activation Test	                        00:51.5
    arm64	        SIGMOID Activation Test	                        00:51.6
    arm64	        Resnet50 Fine Tuning Test	                    06:35.5
    arm64	        MobilenetV2 Fine Tuning Test	                00:56.3
    arm64	        Inception V3 Inference Test (Batch size = 4)	05:20.9
    arm64	        Resnet 50 Inference Test (Batch size = 4)	    08:49.6
    arm64	        EfficientNet-B0 Inference Test (Batch size = 4)	01:44.8
    arm64	        MobilenetV2 Inference Test (Batch size = 4)	    01:13.0
    arm64	        Xception Inference Test (Batch size = 4)	    03:21.5
