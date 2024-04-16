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
pip-upgrade upgrade.txt --skip-package-installation
```