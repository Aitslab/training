# How to set up tensorflow GPU on your windows computer:

CUDA: Framework for NVIDIA GPUs  
CUDNN: Additional packages that enable GPU use  

***IMPORTANT***
It is absolutely essential to match the tensorflow version + CUDA toolkit version + CUDNN version. If the versions do not match, you will get errors.   

Visit https://www.tensorflow.org/install/gpu and https://developer.nvidia.com/cuda-gpus for compatibility checks. 

For my computer with RTX 2060 notebook, CUDA 11.0 + CUDNN 8.0.5 + tensorflow 2.4.0 combination worked.

Remember, higher versions of CUDA allows lower version enabled GPUs to run. So CUDA 11 can run a CUDA 7 enabled GPU.


### Follow these steps

1. If you don't have already, get visual studio express at https://visualstudio.microsoft.com/vs/express/
2. Find and download the CUDA toolkit version that matches your tensorflow version at https://www.tensorflow.org/install/gpu
3. Check the corresponding CUDNN. You need to register for it but it's free and easy.
4. Download the CUDNN zip file and copy the files into your CUDA toolkit folder (all files in bin->bin, lib->lib, include->include)
5. A restart of your notebook is required for the changes to take effect. 
6. Make sure the CUDA path includes the correct path.

Also, for pytorch the cuda toolkit is required to be installed

```pip install cudatoolkit```

### Quick test
Check the following:

```python
import tensorflow as tf
tf.config.list_physical_devices()
```

If you see GPU listed, then you're good. If you see only CPU, you did something wrong.

### Proper test 


```python
import tensorflow as tf
tf.__version__

# check if the gpu is detected
tf.config.experimental.list_physical_devices()

# do a simple calculation
with tf.device("/gpu:0"):
    m = tf.matmul(tf.constant([[1,2,3,4]]), tf.constant([[1],[1],[1],[1]]))

```

If there is a memory error when you do ML tasks, it happens because memory growth isn't enabled. It is required to do GPU tasks.

Add the following lines in the begining of your python script:

```python
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

You can always find memory usage information by opening up a command prompt and typing in ```nvidia-smi```
