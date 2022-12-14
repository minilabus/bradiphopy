import glob
from setuptools import setup

opts = dict(scripts=glob.glob("scripts/*.py"))


if __name__ == '__main__':
    setup(**opts)
