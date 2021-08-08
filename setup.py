import setuptools

# Makes it easy for contributors to install user-facing dependencies.
reqs = []
with open('requirements.txt') as f:
    for line in f:
        if not line.strip().startswith('#'):
            line = line.rstrip('\n')
            reqs.append(line)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aiqc",
    version="3.0.0",#start using double digits.
    author="Layne Sadler",
    author_email="layne.sadler@gmail.com",
    description="End-to-end machine learning on your desktop or server.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://aiqc.readthedocs.io/",
    packages=setuptools.find_packages(),
    include_package_data=True,# Triggers `MANIFEST.in` file.
    python_requires='>=3.5, <=3.8.7', # (tf req Py3.5-3.8)
    license='BSD 3-Clause',
    # Version operands https://pip.pypa.io/en/stable/reference/pip_install/#requirement-specifiers
    # According to Python slack wizards, despite wheel-related warnings when installing aiqc on 
    # a fresh python env, I don't need to require users to install 'wheel'.
    install_requires=reqs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
