import setuptools

with open('README.md', 'r') as handle:
    long_description = handle.read()

setuptools.setup(
    name='geo_passage_retrieval',
    version='1.0.0',
    author='João Coelho',
    author_email='joao.vares.coelho@tecnico.ulisboa.pt',
    description='Geographic Passage Retrieval - Model training and evaluation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'torch',
        'sentence-transformers==1.2.0',
        'transformers',
        #'torchsort',
        'pandas',
        'numpy'
    ],
    python_requires='>=3',
)
