# NBDoc Guide

> "Research is what I'm doing when I don't know what I'm doing." — Wernher von Braun

<!-- Relocated from the legacy `documentation/` directory to the MkDocs hierarchy. -->

This guide explains how to use **NBDoc** to document Jupyter notebooks in this project.

## What is NBDoc?

NBDoc is a tool for generating documentation from Jupyter notebooks. It allows you to:

1. Include notebooks as part of your project documentation
2. Extract specific sections of notebooks for documentation
3. Generate API documentation from notebook code cells
4. Create a searchable documentation site that includes notebook content

## NBDoc Syntax

NBDoc uses HTML comments with special tags to mark sections of your notebook for documentation. These comments are invisible when viewing the notebook normally but are processed when generating documentation.

### Basic Metadata Tags

Place these at the top of your notebook in a markdown cell:

```markdown
<!-- #nbdoc:title Your Notebook Title -->
<!-- #nbdoc:description A brief description of what this notebook does -->
<!-- #nbdoc:version 0.1.0 -->
<!-- #nbdoc:author Your Name -->
<!-- #nbdoc:keywords keyword1, keyword2, keyword3 -->
```

### Section Tags

Use these to mark sections of your notebook:

```markdown
<!-- #nbdoc:section Section Name -->
<!-- #nbdoc:description Description of this section -->
```

You can add these tags to the metadata of a code cell:

```python
# Code cell with section metadata
```

### Code Documentation

NBDoc can extract docstrings and function signatures from your notebook code cells:

```python
def my_function(param1, param2):
    """Return the sum of two numbers."""
    return param1 + param2
```

## Using NBDoc in This Project

### Installation

```bash
uv pip install nbdoc
```

### Generating Documentation

```bash
python -m nbdoc build notebooks/ -o docs/generated/notebooks
```

### MkDocs Integration

```yaml
plugins:
  - nbdoc:
      notebooks_dir: notebooks/
      include_source: true
```

## Best Practices

1. **Keep notebooks clean:** strip debug output before committing.
2. **Use consistent sections:** e.g. "Data Loading", "Training".
3. **Add context & examples.**

## Further Resources

* <https://github.com/nbdoc/nbdoc>
* <https://nbdoc.readthedocs.io/>
