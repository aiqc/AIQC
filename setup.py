import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aiqc",
    version="1.0.72",
    author="Layne Sadler",
    author_email="layne.sadler@gmail.com",
    description="End-to-end machine learning on your desktop or server.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://aiqc.readthedocs.io/",
    packages=setuptools.find_packages(),
    include_package_data=True,# Triggers `MANIFEST.in` file.
    python_requires='>=3.5, <=3.8.7', # (tf req Py3.5-3.8)
    license='AGPLv3',
    # Version operands https://pip.pypa.io/en/stable/reference/pip_install/#requirement-specifiers
    # According to Python slack wizards, despite wheel-related warnings when installing aiqc on 
    # a fresh python env, I don't need to require users to install 'wheel'.
    install_requires=[
        # Mandatory versions:
        'tensorflow==2.4.1'#tensorflow.org/install/pip
        , 'Keras==2.4.3'#https://docs.floydhub.com/guides/environments/
        , 'h5py==2.10.0'#(tf2.4.1 req h5py~=2.10.0)
        # Frameworks:
        , 'peewee==3.14.3'#Just pip show'ed where this was at during aiqc 1.0.0
        , 'scikit-learn==0.24.1'#Just pip show'ed where this was at during aiqc 1.0.0
        , 'numpy==1.19.5'# The version required by tensorflow. Wasn't accepting 1.20.
        , 'pandas==1.2.3'# Heavily depended on so other packages will install it.
        , 'Pillow==8.1.2'
        , 'pyarrow==3.0.0'
        , 'plotly==4.14.3'
        # Programmatic:
        , 'appdirs==1.4.4'
        , 'natsort==7.1.1'
        , 'tqdm==4.59.0'
        , 'validators==0.18.2'
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
