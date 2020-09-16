# Jack C. Cook
# Tuesday, September 8, 2020

from setuptools import setup


def get_reqs(fname):
    """
    Get the requirements list from the text file
    JCC 03.10.2020
    :param fname: the name of the requirements text file
    :return: a list of requirements
    """
    file = open(fname)
    data = file.readlines()
    file.close()
    return [data[i].replace('\n', '') for i in range(len(data))]


setup(name='MachineLearning',
      version='0.0.1',
      packages=['MachineLearning'],
      install_requires=get_reqs('requirements.txt'),
      author='Jack C. Cook',
      author_email='jack.cook@okstate.edu',
      description='A package containing the scripts developed in CS 5783 at Oklahoma State University.')
