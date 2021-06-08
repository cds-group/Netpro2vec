from setuptools import setup

setup(
   name='Netpro2vec',
   version='0.1.0',
   author='Ichcha Manipur and Maurizio Giordano',
   author_email='ichcha.manipur@icar.cnr.it',
   packages=['netpro2vec'],
   #scripts=['bin/script1','bin/script2'],
   #url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='The netpro2vec graph-embedding method',
   long_description=open('README.txt').read(),
   install_requires=[
      "tqdm>=4.46.1",
      "pandas>=1.0.2",
      "numpy>=1.16.2",
      "gensim<=3.8.3",
      "scipy>=1.4.1",
      "joblib>=0.14.1",
      "python_igraph>=0.8.2"
   ],
)
