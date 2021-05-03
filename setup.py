import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="symbolic-dynamics-jzc", 
    version="0.0.1",
    author="Justin Cai",
    author_email="jc@justincai.com",
    description="A symbolic dynamics package for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jzc/symbolic-dynamics",
    packages=["symbolic_dynamics"],
    python_requires=">=3.9",
    install_requires=["networkx"],
)
