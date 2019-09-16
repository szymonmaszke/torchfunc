import setuptools

name, version = open("METADATA").read().splitlines()

setuptools.setup(
    name=name,
    version=version,
    license="MIT",
    author="Szymon Maszke",
    author_email="szymon.maszke@protonmail.com",
    description="PyTorch functions to improve performance, analyse models and make your life easier.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/szymonmaszke/torchfunc",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=open("environments/requirements.txt").read().splitlines(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
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
