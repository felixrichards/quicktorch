from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="quicktorch",
    version="0.1",
    description="Simple pytorch wrapper.",
    long_description=long_description,
    author="felixajwndqw",
    url="github.com/felixajwndqw/quicktorch",
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scikit-image', 'sklearn',
                      'torch', 'torchvision']
)