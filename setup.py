import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="symbolic-dynamics",
    version="0.1.1",
    author="Justin Cai",
    author_email="jc@justincai.com",
    description="A symbolic dynamics package for Python",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jzc/symbolic-dynamics",
    packages=["symbolic_dynamics"],
    package_data={"symbolic_dynamics": ["tests/*.py"]},
    python_requires=">=3.8",
    install_requires=["networkx>=2.5"],
)
