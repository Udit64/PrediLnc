from setuptools import setup, find_packages

setup(
    name='your_project_name',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'Levenshtein==0.23.0',
        'numpy==1.23.5',
        'pandas==2.2.0',
        'requests==2.28.1',
        'torch==2.20.0',
        'tensorflow==2.9.0',
        'keras==2.9.0',
        'BeautifulSoup==5.7.1',
        'flask==3.0.0',
        'biopython==1.5.9'  # Note: Bio is typically installed as biopython
    ],
)
