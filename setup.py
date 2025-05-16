import setuptools


__version__ = "0.0.1"

SRC_REPO = "CorrectiveRAG"
AUTHOR_USER_NAME = "FraidoonOmarzai"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="Implementation of Corrective RAG based on research paper",
    packages=setuptools.find_packages()
)