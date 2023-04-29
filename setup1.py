import setuptools

setuptools.setup(
    name="torchdiffeq",
    version="0.0.1",
    author="Ting Dang",
    author_email="td464@cam.ac.uk",
    description="A novel contrained neural ordinary differential equations",
    packages=['torchdiffeq', 'torchdiffeq._impl'],
    install_requires=['torch>=0.4.1'],
    classifiers=( "Programming Language :: Python :: 3"),)
