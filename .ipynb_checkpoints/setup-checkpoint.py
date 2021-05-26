# from os import name
# import setuptools

# with open("README.md", 'r') as fh:
#     long_description = fh.read()
    
# setuptools.setup(
#     name= titanic-survive-prediction,
#     version=0.0.1,
#     author=nnagita,
#     author_email=nsagita47@gmail.com,
#     description=exercise project, 
#     long_description=long_description, 
#     packages=setuptools.find_packages(), 
#     classifiers=[
#         " programming language :: Python :: 3"
#     ], 
#     install_requires=[
#         "matplotlib==3.3.2",
#         "numpy==1.19.4",
#         "pandas==1.1.4",
#         "pydantic==1.6.1",
#         "seaborn==0.11.0"
 
#     ],
#     python_requires = ">=3.8"

# )

from os import name
import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="titanic-survived-prediction",
    version="0.0.1",
    author="Aris B W",
    author_email="arisbuw@gmail.com",
    description="Exercise project",
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    install_requires=[
        "matplotlib==3.3.2",
        "numpy==1.19.4",
        "pandas==1.1.4",
        "pydantic==1.6.1",
        "seaborn==0.11.0",
        "scikit-learn==0.24"
    ],
    python_requires = ">=3.7"
)
