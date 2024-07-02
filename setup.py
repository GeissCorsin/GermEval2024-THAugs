from setuptools import setup, find_packages

setup(
    name='GermEval2024_THAugs',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
