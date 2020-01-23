import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tlopu",
    version="0.0.1",
    author="Giuseppe Luca Tommasone",
    author_email="luca@lighton.io",
    description="Blogpost code for TL on videos.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LucaT1995/TL_blogpost",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scikit-learn", "scipy", "torch", "torchvision", "pandas", "argparse", "tqdm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)