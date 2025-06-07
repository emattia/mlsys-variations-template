# NBDoc Guide

This guide explains how to use NBDoc to document Jupyter notebooks in this project.

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
    """
    This function does something useful.

    Args:
        param1: The first parameter
        param2: The second parameter

    Returns:
        The result of the computation
    """
    return param1 + param2
```

## Using NBDoc in This Project

### Installation

NBDoc is included in the project dependencies. If you need to install it separately:

```bash
uv pip install nbdoc
```

### Generating Documentation

To generate documentation from notebooks:

```bash
python -m nbdoc build notebooks/ -o documentation/generated/notebooks
```

### Integrating with Other Documentation

NBDoc can be integrated with other documentation tools like MkDocs:

```yaml
# mkdocs.yml
plugins:
  - nbdoc:
      notebooks_dir: notebooks/
      include_source: true
```

## Best Practices

1. **Keep notebooks clean**: Remove unnecessary output and debug cells before documenting
2. **Use consistent section names**: Standardize section names across notebooks (e.g., "Data Loading", "Model Training")
3. **Add context**: Explain the purpose and assumptions of each code cell
4. **Include examples**: Show example usage and expected outputs
5. **Document parameters**: Clearly document function parameters and return values
6. **Add visualizations**: Include relevant plots and diagrams to illustrate concepts

## Example

See `notebooks/example_analysis.ipynb` for a complete example of a documented notebook.

## NBDoc Configuration

You can configure NBDoc by creating a `nbdoc_config.json` file in the project root:

```json
{
  "project_name": "Analysis Template",
  "output_dir": "documentation/generated",
  "exclude_patterns": ["*draft*", "*scratch*"],
  "include_source": true,
  "theme": "readthedocs"
}
```

## Generating API Documentation

To generate API documentation from notebooks:

```bash
python -m nbdoc api notebooks/ -o documentation/generated/api
```

This will extract all functions and classes defined in notebooks and generate API documentation.

## Further Resources

- [NBDoc GitHub Repository](https://github.com/nbdoc/nbdoc)
- [NBDoc Documentation](https://nbdoc.readthedocs.io/)
