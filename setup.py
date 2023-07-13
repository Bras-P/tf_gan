from setuptools import setup, find_packages

setup(
    name='tf_gan',
    packages=[
        'tf_gan',
    ],
    url='',
    author="Pierre Bras",
    description='An implementation of GANs in TensorFlow',
    long_description=open('README.md').read(),
    # install_requires=[
    #     "tensorflow==2.10.0",
    #     ],
    include_package_data=True,
    license='MIT',
)