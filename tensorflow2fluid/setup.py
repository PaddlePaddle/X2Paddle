from __future__ import absolute_import
from setuptools import setup, find_packages
from io import open

setup(
    name='tensorflow2fluid',

    version='0.0.1',

    description='Translate TensorFlow Model to PaddlePaddle',
    url='http://paddlepaddle.org',
    author = 'PaddlePaddle Development Group',
    author_email='paddle-dev@baidu.com',
    license='Apache 2',
    packages=find_packages(),

    install_requires=[
        'paddlepaddle >= 1.2.1',
        'tensorflow >= 1.12.0',
    ],

    entry_points={
        'console_scripts': [
            'tf2fluid = src.convert:_main',
        ],
    },
)
