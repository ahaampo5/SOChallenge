#nsml: nvcr.io/nvidia/pytorch:20.12-py3

from distutils.core import setup

setup(
    name='NSML Small Object train example',
    version='1.1',
    install_requires=[
        'albumentations==0.5.2',
        'opencv-python==4.1.2.30',
        'ensemble-boxes'
    ]
)