import pathlib

import setuptools


def read(HERE: pathlib.Path, filename, variable):
    namespace = {}

    exec(open(HERE / "torchfunc" / filename).read(), namespace)  # get version
    return namespace[variable]


HERE = pathlib.Path(__file__).resolve().parent

setuptools.setup(
    name=read(HERE, pathlib.Path("_name.py"), "_name"),
    version=read(HERE, pathlib.Path("_version.py"), "__version__"),
    license="MIT",
    author="Szymon Maszke",
    author_email="szymon.maszke@protonmail.com",
    description="PyTorch functions to improve performance, analyse models and make your life easier.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/szymonmaszke/torchfunc",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=open("environments/requirements.txt").read().splitlines(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Website": "https://szymonmaszke.github.io/torchfunc",
        "Documentation": "https://szymonmaszke.github.io/torchfunc/#torchfunc",
        "Issues": "https://github.com/szymonmaszke/torchfunc/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc",
    },
    keywords="pytorch torch functions performance visualize utils utilities recording",
)
