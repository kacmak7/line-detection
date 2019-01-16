import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
        name='line_detection',
        version='0.1',
        author='Kacper Makuch',
        author_email='kacpermakuch7@gmail.com',
        description='Road lanes detector',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/kacmak7/line-detection.git',
        packages=setuptools.find_packages(),
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent',
        ],
)
