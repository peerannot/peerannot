from setuptools import setup, find_packages

setup(
    name="PeerAnnot",
    version="0.0.1-2",
    description="Crowdsourcing library",
    author="Contributors",
    author_email="tanguy.lefort@umontpellier.fr",
    url="https://peerannot.github.io/peerannot/",
    packages=find_packages(),
    long_description=open("README.md").read(),
)
