# PyPI Package

### Dependencies for build process.

```bash
pip3 install --upgrade wheel twine
```

---

### Steps to build & upload.

```bash
cd ~/Desktop/AIQC

python3 setup.py sdist bdist_wheel

python3 -m twine upload --repository pypi dist/*
# username: __token__
# password: <paste in the token>

# Delete build-generated files before git commit.
rm -r build dist aiqc.egg-info

# Within `setup.py` proactively update the minor version number for next time.
```

---

### Testing a Fresh Install

# Uninstalls all user-installed packages.
```bash
pip freeze | xargs pip uninstall -y
pip install --upgrade aiqc
```