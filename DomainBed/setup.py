from setuptools import setup, find_packages

setup(
    name='domainbed',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.3',
        'wilds>=1.2.2',
        'imageio>=2.9.0',
        'gdown>=3.13.0',
        # 'torch==1.7.1',
        # 'torchvision>=0.8.2',
        'tqdm>=4.62.2',
        'backpack>=0.1',
        'parameterized>=0.8.1',
        'Pillow>=8.3.2',
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
