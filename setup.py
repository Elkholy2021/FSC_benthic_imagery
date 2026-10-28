from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="FSCBI", 
    version="0.1.0",
    author="Hassan Elkholy",
    author_email="hassan.elkholy@ntnu.no",
    description="Few-Shot Classification for Benthic Imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Elkholy2021/FSC_benthic_imagery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Add your package dependencies here
        # "numpy>=1.20.0",
        # "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
)