# PyPI Package

### Dependencies for build process.

Install pip
```bash
pip install --upgrade pip
```
or
```bash
conda install pip
```

```bash
pip install --upgrade wheel
pip install --upgrade aiqc
```

---

### Steps to build & upload.

```bash
cd ~/Desktop/AIQC

# Make sure the build files aren't hanging around from previous attempt.
rm -r build dist aiqc.egg-info

# Make sure to update version number in `setup.py` first.
# Otherwise you need to run the `rm` line again.
python3 setup.py sdist bdist_wheel

python3 -m twine upload --repository pypi dist/*
# username: __token__
# password: <paste in the token>

# If "File already exists" error, then run `rm` below & update version above.
# Delete build-generated files before git commit.
rm -r build dist aiqc.egg-info
```

---

### Testing a Fresh Install

# Uninstalls all user-installed packages.
```bash
pip freeze | xargs pip uninstall -y
pip install --upgrade aiqc
```