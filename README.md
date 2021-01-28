![AIQC (wide)](/docs/images/aiqc_logo_wide_black_docs.png)

---
*pre-alpha; in active development*

# Documentation

[ReadTheDocs](https://aiqc.readthedocs.io/)


# PyPI Package

### Steps to Build & Upload

```bash
pip3 install --upgrade wheel twine
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository pypi dist/*
# username: __token__
# password: <paste in the token>
rm -r build dist aiqc.egg-info
# proactively update the version number in setup.py next time
```