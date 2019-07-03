import glob
import os

from collections import defaultdict

examples_page = """
.. examples:

Examples
--------------------------

We developed scikit-quantum using a simple API inspired by scikit-learn. If you want a specific example, feel free to file an issue.
"""

examples_by_folder = defaultdict(list)
for filename in glob.iglob('examples/**/*.py', recursive=True):
     f = open(filename, "r")
     source_code = f.read()
     f.close()

     occurence_of_first_triple_quotes = source_code.find('"""')
     occurence_of_second_triple_quotes = source_code[occurence_of_first_triple_quotes + 3:].find('"""') + 3
     source_code_description = source_code[occurence_of_first_triple_quotes + 3: occurence_of_second_triple_quotes]
     actual_source_code = source_code[occurence_of_second_triple_quotes + 3:]
     folder_name = filename.split("/")[1]

     examples_by_folder[folder_name].append({"filename": filename,
										"source_code_description": source_code_description,
										"actual_source_code": actual_source_code})

os.system("rm -rf docs_source/examples")
os.system("mkdir docs_source/examples")
for folder_name, examples in examples_by_folder.items():

	list_of_examples = ".. toctree::\n\t:maxdepth: 2\n\n"
	
	for e in examples:
		rst_filename = e["filename"].replace("/", "_").replace(".py", ".rst")

		f = open("docs_source/examples/%s" % rst_filename, "w")
		example_rst = e["source_code_description"] + "\n.. code-block::\n\n" + "\t".join(e["actual_source_code"].splitlines(True))
		f.write(example_rst)
		f.close()

		list_of_examples += "\t" + "examples/" + rst_filename + "\n"

	examples_page += "\n%s\n=========================================================================\n\n%s" % (folder_name.replace("_", " ").title(), list_of_examples)


f = open("docs_source/examples.rst", "w")
f.write(examples_page)
f.close()
