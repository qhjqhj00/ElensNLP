from setuptools import setup, find_packages

import os

CACHE_ROOT = os.path.expanduser(os.path.join('~', '.lensnlp'))

if not os.path.exists(CACHE_ROOT):
    os.mkdir(CACHE_ROOT)

setup(
    name="lensnlp",
    version="2019.10.17",
    description="distributed by Elensdata",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Hongjin Qian",
    author_email="tommy@chien.io",
    url="https://www.elensdata.com/",
    packages=find_packages(exclude="test"),  # same as name
    license="MIT",
    install_requires=[
        "torch>=1.0.0",
        "gensim>=3.4.0",
        "tqdm>=4.26.0",
        "segtok>=1.5.7",
        "matplotlib>=2.2.3",
        "mpld3>=0.3",
        "sklearn",
        "langid",
        "transformers",
        "sqlitedict>=1.6.0",
        "deprecated>=1.2.4",
        "hyperopt>=0.1.1",
        "bpemb>=0.2.9",
        "regex>=2018.1.10",
        "pypinyin>=0.35.0",
        "langid",
        "jieba"
    ],
    include_package_data=True,
    python_requires=">=3.6",
)
