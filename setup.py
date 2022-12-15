from setuptools import setup, find_packages
from peerannot import __version__

setup(
    name="PeerAnnot",
    version=__version__,
    description="Crowdsourcing library",
    author="Contributors",
    author_email="tanguy.lefort@umontpellier.fr",
    url="https://peerannot.github.io/peerannot/",
    packages=find_packages(),
    long_description=open("README.md").read(),
    entry_points={
        "console_scripts": ["peerannot = peerannot.runners:peerannot"]
    },
)
