```bash
cd pypto
git remote add fork-gbisbas https://github.com/georgebisbas/pypto.git
git fetch fork-gbisbas

git config --global user.name "georgebisbas"
git config --global user.email "georgios.bismpas@h-partners.com"
git config --global pull.rebase true

export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so

pip install --no-build-isolation -e .

```


simpler: 
```bash
cd simpler
git remote add fork-gbisbas https://github.com/georgebisbas/simpler.git
git fetch fork-gbisbas

git config --global user.name "georgebisbas"
git config --global user.email "georgios.bismpas@h-partners.com"
git config --global pull.rebase true

export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so

pip install --no-build-isolation -e .

```

```
git pull
cmake --build build --parallel
pip install --no-build-isolation -e .
```


```
cd /opt/pypto
npu-smi info
python -c "import pypto; print('pypto ok')"
which ptoas; ptoas --version

cd /opt/pypto
git remote add fork-gbisbas https://github.com/georgebisbas/pypto.git
git fetch fork-gbisbas

git config --global user.name "georgebisbas"
git config --global user.email "georgios.bismpas@h-partners.com"
git config --global pull.rebase true

cd /opt/
export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so
git clone https://github.com/hw-native-sys/pypto-lib.git
git clone https://github.com/hw-native-sys/simpler.git
git clone https://github.com/hw-native-sys/pto-isa.git
git clone https://github.com/georgebisbas/pypto-tooling.git
git clone https://github.com/georgebisbas/pypto-profiling.git
git clone https://github.com/hw-native-sys/PTOAS
```