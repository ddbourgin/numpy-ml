# flake8: noqa
from codecs import open

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt") as requirements:
    REQUIREMENTS = [r.strip() for r in requirements if r != "\n"]

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/ddbourgin/numpy-ml/issues",
    "Documentation": "https://numpy-ml.readthedocs.io/en/latest/",
    "Source": "https://github.com/ddbourgin/numpy-ml",
}

setup(
    name="numpy-ml",
    version="0.1.2",
    author="David Bourgin",
    author_email="ddbourgin@gmail.com",
    project_urls=PROJECT_URLS,
    url="https://github.com/ddbourgin/numpy-ml",
    description="Machine learning in NumPy",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    install_requires=REQUIREMENTS,
    packages=find_packages(),
    license="GPLv3+",
    include_package_data=True,
    python_requires=">=3.5",
    extras_require={"rl": ["gym", "matplotlib"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
)
