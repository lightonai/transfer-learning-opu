import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tlopu",
    version="0.0.1",
    author="Giuseppe Luca Tommasone",
    author_email="luca@lighton.io",
    description="Blogpost code for Transfer Learning on videos.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LucaT1995/TL_blogpost",
    packages=setuptools.find_packages(),
    install_requires=["numpy==1.17.5", "scikit-learn==0.22.1", "scipy==1.4.1", "torch==1.2", "torchvision==0.4",
                      "pandas==0.24.2", "tqdm==4.41.1", "matplotlib==3.0.2", "Pillow==6.2.0", "lightonml==1.0.2",
                      "lightonopu==1.0.4"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)