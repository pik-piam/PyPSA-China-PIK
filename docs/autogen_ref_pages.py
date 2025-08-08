from pathlib import Path

import mkdocs_gen_files

# Set the root and scripts directory
root = Path(__file__).parent.parent
docu_dir = root / "docs"
scripts_dir = root / "workflow" / "scripts"

# Loop through all Python files in the `scripts_dir` using rglob
for path in sorted(scripts_dir.rglob("[!_]*.py")):
    # Generate the module path relative to the scripts directory, excluding the '.py' extension
    module_path = path.relative_to(scripts_dir).with_suffix("")

    # Generate the corresponding markdown file path, replacing '.py' with '.md'
    doc_path = path.relative_to(scripts_dir).with_suffix(".md")

    # Final path for the documentation, placed in the `reference` folder
    full_doc_path = Path("reference/", doc_path)

    # Extract module parts
    parts = tuple(module_path.parts)

    # Remove '__init__' or '__main__' from module path if present
    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue  # Skip '__main__' files as they don't need documentation

    # Create the documentation file
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(
            f"::: {identifier}", file=fd
        )  # MkDocs-specific syntax to pull in the module

    # Set the edit path to link the documentation back to the original Python file
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


# # Create the navigation structure
def make_yaml_nav():
    # Create the navigation structure
    nav = mkdocs_gen_files.Nav()

    for path in sorted(scripts_dir.rglob("[!_]*.py")):
        module_path = path.relative_to(scripts_dir).with_suffix("")
        doc_path = path.relative_to(scripts_dir).with_suffix(".md")

        parts = tuple(module_path.parts)
        if parts[-1] in {"__init__", "__main__"}:
            continue

        full_doc_path = doc_path  # Fix to remove 'reference' prefix
        nav[parts] = full_doc_path

        return nav


nav = make_yaml_nav()
with mkdocs_gen_files.open("reference_nav.yml", "w") as nav_file:
    nav_file.write("Reference:\n")  # Start the "Reference" section
    nav_items = {itm.title: itm.filename for itm in nav.items()}
    for title, value in nav_items.items():
        nav_file.write(f"- {value}\n")  # Write each file path


def make_literate_nav():
    """Make a literate-nav style nav for mkdocs. The nav of mkdocs.yml then should point to the corresponding file"""
    # Create the navigation structure
    nav = mkdocs_gen_files.Nav()

    for path in sorted(scripts_dir.rglob("[!_]*.py")):
        module_path = path.relative_to(scripts_dir).with_suffix("")
        doc_path = path.relative_to(scripts_dir).with_suffix(".md")

        parts = tuple(module_path.parts)
        if parts[-1] in {"__init__", "__main__"}:
            continue
        parts = [f"    - {part} " for part in parts]  # Add space to each part

        nav[parts] = doc_path  # Keep this for nav structure building
    return nav


def write_literate_nav(
    lit_nav: mkdocs_gen_files.Nav,
    prepend: list[str],
    fname="reference_nav.yml",
):
    """Write the literate-nav style nav for mkdocs. The nav of mkdocs.yml then should point to the corresponding fil
    Args:
        lit_nav: mkdocs_gen_files.Nav from make_literate_nav
        prepend: list of strings to prepend to the nav file (other sections)
        fname: name of the file to write to (yaml)
    """
    with mkdocs_gen_files.open(fname, "w") as nav_file:
        for line in prepend:
            if not line.endswith("\n"):
                line += "\n"
            nav_file.write(line)
        nav_file.writelines(lit_nav)
