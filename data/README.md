# Data Directory

This directory contains all data files used in the project.

## Structure

- **raw/**: Original, immutable data dumps. This data should never be modified.
- **processed/**: Cleaned and processed data, ready for analysis.
- **interim/**: Intermediate data that has been transformed but is not yet in its final form.
- **external/**: Data from external sources, such as third-party datasets.

## Usage Guidelines

1. **Never modify raw data**: Always keep the original data intact for reproducibility.
2. **Document data sources**: Include information about where data came from, when it was collected, and any relevant context.
3. **Version control**: For large data files, consider using Git LFS or DVC.
4. **Data formats**: Prefer efficient formats like Parquet for large datasets.

## Data Dictionary

For each dataset, consider creating a data dictionary that describes:

- Column names and descriptions
- Data types
- Units of measurement
- Valid ranges or categories
- Missing value codes

Example:

| Column Name | Description | Data Type | Units | Valid Range |
|-------------|-------------|-----------|-------|-------------|
| temperature | Daily temperature | float | Celsius | -50 to 50 |
| precipitation | Daily rainfall | float | mm | 0 to 500 |
| location_id | Weather station ID | int | N/A | 1 to 1000 |

## Data Processing Workflows

Data processing workflows are defined in the `workflows/` directory. These workflows transform raw data into processed data, and should be reproducible and well-documented.

## Large Files

Large data files should not be committed to Git. Instead:

1. Add them to `.gitignore`
2. Consider using Git LFS or DVC for version control
3. Document how to obtain or generate these files

## Data Privacy

Be mindful of data privacy concerns:

1. Do not commit sensitive or personal data to the repository
2. Anonymize data when necessary
3. Include appropriate data usage disclaimers
