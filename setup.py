from setuptools import setup, find_packages

setup(
    name="itantra",
    version="1.0.0",
    package_dir={"": "src", "benchmark": "benchmark", "dataset": "dataset", "ui": "ui"},
    packages=find_packages(where="src") + ["benchmark", "dataset", "ui"],
)
