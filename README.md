# DLAMP

# set environment
activate a virtual envirionment, here use conda env
```
conda create --name [env name] python=3.11 -y
conda activate [env name]
```
install NVIDIA modulus package
```
git clone git@github.com:NVIDIA/modulus.git && cd modulus
make install
```
other packages
```
pip3 install -r requirements.txt
pip3 install hydra-core --upgrade
```
install onnxruntime according to your CUDA version, please check [onnxruntime_official](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime) for more details.
```
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```