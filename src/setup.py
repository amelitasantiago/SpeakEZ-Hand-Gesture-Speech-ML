#pip install -e
from setuptools import setup, find_packages

setup(
    name="speakez",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[l.strip() for l in open("requirements.txt")],
)
