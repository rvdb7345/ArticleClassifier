# Project Installation

The project environment may be set up using:
```conda create --name articleclassifier --file requirements.txt```

Confirm afterwards that the GPU configuration has succeeded (if actually using Nvidia GPUs, but if not, you should get
trained models from someone, because training without GPUs takes ages.)
You can do a quick check by looking at the output of the following code:

```python
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION', )
from subprocess import call
# call(["nvcc", "--version"]) does not work
! nvcc - -version
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
# call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())
print('Available devices ', torch.cuda.device_count())
```

