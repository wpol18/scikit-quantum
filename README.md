
![scikit-quantum logo](https://scikit-quantum.com/html/_static/logo.png)

# scikit-quantum

scikit-quantum is a Python module for quantum computing and quantum machine learning built primarily on top of PennyLane and is distributed under the Apache 2.0 license.

scikit-quantum is aimed to provide a user-friendly API while still harnessing the power of a quantum device. The goal is to create a set of tools that are vendor agnostic (although that has limited feasibility at this time), and bring the power of several quantum algorithms to new users.

# Building the Documentation

We use GitHub pages to host our documentation. 

The user-guide and manual for scikit-quantum can be found at: https://scikit-quantum.com

To compile the docs to the `docs` folder run:

> make -f build_tools/build_docs/Makefile html
