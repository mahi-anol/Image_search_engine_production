from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This is a Image Search Engine which utilizes MobileNetV2 as backbone for Embedding and for now utilizes faiss indexing for Vector search....',
    author='Mahi_SSL',
    license='MIT',
)
