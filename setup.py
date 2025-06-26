from setuptools import setup, find_packages

setup(
    name="akorn",
    version="0.1",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
)
