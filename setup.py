import setuptools

setuptools.setup(
    name="x2paddle",
    version="0.4.0",
    author="dltp-sz",
    author_email="dltp-sz@baidu.com",
    description=
    "a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks.",
    long_description_content_type="text/markdown",
    url="https://github.com/PaddlePaddle/x2paddle",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={'console_scripts': ['x2paddle=x2paddle.convert:main']})
