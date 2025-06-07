# Documentation Directory

This directory contains project documentation, guides, and references.

## Contents

- **nbdoc_guide.md**: Guide for using NBDoc to document Jupyter notebooks
- **generated/**: Auto-generated documentation (not committed to Git)
  - **notebooks/**: Documentation generated from notebooks
  - **api/**: API documentation generated from source code

## Documentation Standards

### General Guidelines

1. **Keep documentation up-to-date**: Update documentation when code changes
2. **Use clear language**: Write in simple, concise language
3. **Include examples**: Provide code examples for key concepts
4. **Link related documents**: Cross-reference related documentation
5. **Version documentation**: Ensure documentation matches the code version

### Markdown Style

- Use ATX-style headers (`#` for titles, `##` for sections)
- Use fenced code blocks with language specifiers (```python)
- Use bullet points for lists
- Use tables for structured data
- Include a table of contents for longer documents

## Generating Documentation

### Notebook Documentation

To generate documentation from notebooks using NBDoc:

```bash
python -m nbdoc build notebooks/ -o documentation/generated/notebooks
```

See `nbdoc_guide.md` for more details on using NBDoc.

### API Documentation

To generate API documentation from source code:

```bash
python -m pdoc --html --output-dir documentation/generated/api src
```

## Documentation Tools

This project uses the following documentation tools:

- **Markdown**: For general documentation
- **NBDoc**: For notebook documentation
- **pdoc**: For API documentation
- **MkDocs** (optional): For creating a documentation website

## Setting Up a Documentation Website

To create a documentation website with MkDocs:

1. Install MkDocs:
   ```bash
   uv pip install mkdocs mkdocs-material
   ```

2. Create a new MkDocs project:
   ```bash
   mkdocs new .
   ```

3. Configure MkDocs in `mkdocs.yml`:
   ```yaml
   site_name: Analysis Template
   theme: material
   docs_dir: documentation
   ```

4. Build and serve the documentation:
   ```bash
   mkdocs serve
   ```

## Best Practices

1. **Document as you code**: Write documentation alongside code
2. **Review documentation**: Include documentation in code reviews
3. **Test documentation**: Ensure code examples work
4. **Get feedback**: Ask others to review documentation for clarity
5. **Use diagrams**: Include visual aids for complex concepts
