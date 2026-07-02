```bash
git remote add fork-gbisbas https://github.com/georgebisbas/pypto.git
git fetch fork-gbisbas

git config --global user.name "georgebisbas"
git config --global user.email "georgios.bismpas@h-partners.com"
git config --global pull.rebase true

pip install --no-build-isolation -e .

export LD_PRELOAD=${CANN_HOME}/aarch64-linux/lib64/libhccl.so

```