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