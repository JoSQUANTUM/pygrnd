'''Copyright 2022 JoS QUANTUM GmbH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='pygrnd',
    version='0.1.7',
    description='A python library for quantum algorithms and software',
    long_description_content_type="text/markdown",
    long_description=long_description,
    url='https://github.com/JoSQUANTUM/pygrnd',
    author='JoS QUANTUM',
    author_email='contact@jos-quantum.de',
    license='Apache 2.0',
    zip_safe=False,
    include_package_data=True,
    packages= find_packages(),
    install_requires=['qiskit',
                      'numpy',
                      'dimod',
                      'dwave-greedy',
                      'dwave-neal',
                      'pennylane',
                      'pandas',
                      'seaborn',
                      'tqdm',
                      'pylatexenc',
                      'scikit-learn',
                      'DateTime',
                      'shap',
                      'xgboost',
                      'holidays',
                      'imblearn'
                      ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent'
    ],
)
