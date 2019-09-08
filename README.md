[![Board Status](https://dev.azure.com/matkallada/833af0e0-daeb-435e-b22b-c6e157692e13/c182d7d2-88f9-411a-abfc-2ca79e11ba5d/_apis/work/boardbadge/eb4d3f2e-ffe0-4d9b-81ca-f5e281c52ad1)](https://dev.azure.com/matkallada/833af0e0-daeb-435e-b22b-c6e157692e13/_boards/board/t/c182d7d2-88f9-411a-abfc-2ca79e11ba5d/Microsoft.RequirementCategory)

![scikit-quantum logo](https://scikit-quantum.com/html/_static/logo.png)

# scikit-quantum

scikit-quantum is a Python module for quantum computing and quantum machine learning built primarily on top of PennyLane and is distributed under the Apache 2.0 license.

scikit-quantum is aimed to provide a user-friendly API while still harnessing the power of a quantum device. The goal is to create a set of tools that are vendor agnostic (although that has limited feasibility at this time), and bring the power of several quantum algorithms to new users.

# Building the Documentation

We use GitHub pages to host our documentation. 

The user-guide and manual for scikit-quantum can be found at: https://scikit-quantum.com

To compile the docs to the `docs` folder run:

> make -f build_tools/build_docs/Makefile html
