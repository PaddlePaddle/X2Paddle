import setuptools
import x2paddle

long_description = "X2Paddle is a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks.\n\n"
long_description += "Usage: x2paddle --framework tensorflow --model tf_model.pb --save_dir paddle_model\n"
long_description += "GitHub: https://github.com/PaddlePaddle/X2Paddle\n"
long_description += "Email: dltp-sz@baidu.com"

setuptools.setup(
    name="x2paddle",
    version=x2paddle.__version__,
    author="dltp-sz",
    author_email="dltp-sz@baidu.com",
    description="a toolkit for converting trained model to PaddlePaddle from other deep learning frameworks.",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/PaddlePaddle/x2paddle",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license='Apache 2.0',
    entry_points={'console_scripts': ['x2paddle=x2paddle.convert:main', ]})
