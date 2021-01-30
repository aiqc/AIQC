# PyPI Package

### Steps to Build & Upload

```bash
pip3 install --upgrade wheel twine

python3 setup.py sdist bdist_wheel

python3 -m twine upload --repository pypi dist/*
# username: __token__
# password: <paste in the token>

# Delete build-generated files before git commit.
rm -r build dist aiqc.egg-info

# `setup.py` proactively update the version number for next time, you won't
```