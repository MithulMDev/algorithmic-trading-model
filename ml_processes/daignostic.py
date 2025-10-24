import tensorflow as tf
import os
import ctypes.util

print("=" * 70)
print("CUDA DETECTION DIAGNOSTIC")
print("=" * 70)

print(f"\nTensorFlow: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# Check build info
try:
    from tensorflow.python.platform import build_info
    print(f"\nBuild Info:")
    print(f"  CUDA version: {build_info.build_info.get('cuda_version', 'N/A')}")
    print(f"  cuDNN version: {build_info.build_info.get('cudnn_version', 'N/A')}")
except Exception as e:
    print(f"Could not get build info: {e}")

# Check environment
print(f"\nEnvironment Variables:")
print(f"  CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not set')}")
print(f"  CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"  CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

# Check for CUDA DLLs
print(f"\nLooking for CUDA libraries...")
libs = ['cudart64_110', 'cudart64_11', 'cublas64_11', 'cudnn64_8']
for lib in libs:
    path = ctypes.util.find_library(lib)
    if path:
        print(f"  ✓ {lib}: {path}")
    else:
        print(f"  ✗ {lib}: Not found in PATH")

# Check PATH
conda_bin_in_path = any('conda' in p.lower() and 'bin' in p.lower() for p in os.environ.get('PATH', '').split(';'))
print(f"\nConda bin in PATH: {conda_bin_in_path}")

# Try to load GPU
print(f"\nAttempting to list GPUs...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Found {len(gpus)} GPU(s)!")
else:
    print(f"❌ No GPUs found")

print("=" * 70)