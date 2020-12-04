import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aiqc",
    version="0.0.80",
    author="Layne Sadler",
    author_email="layne@protonmail.com",
    description="End-to-end machine learning on your desktop or server.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aiqc",
    packages=setuptools.find_packages(),
    include_package_data=True,#triggers MANIFEST.in
    python_requires='>=3.7.6',
    license='AGPLv3',
    install_requires=[
        'appdirs',
        'h5py==2.10.0',
        'keras',
        'numpy==1.18.5',
        'pandas',
        'peewee',
        'plotly',
        'pyarrow',
        'scikit-learn',
        'tensorflow',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
