
from setuptools import setup, find_packages
from os import path
import re

package_name="sne4onnx"
root_dir = path.abspath(path.dirname(__file__))

with open("README.md") as f:
    long_description = f.read()

with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

setup(
    name=package_name,
    version=version,
    description=\
        "A very simple tool for situations where optimization with onnx-simplifier "+
        "would exceed the Protocol Buffers upper file size limit of 2GB, "+
        "or simply to separate onnx files to any size you want. Simple Network Extraction for ONNX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Katsuya Hyodo",
    author_email="rmsdh122@yahoo.co.jp",
    url="https://github.com/PINTO0309/sne4onnx",
    license="MIT License",
    packages=find_packages(),
    platforms=["linux", "unix"],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            "sne4onnx=sne4onnx:main"
        ]
    }
)
