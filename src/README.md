# Source Code Reference

Core ML components organized for production deployment, extensibility, and maintainability.

## Module Structure

```
src/
├── api/                    # FastAPI service
├── data/                   # Data processing
├── models/                 # Model training/inference
├── plugins/                # Extensible components
├── config/                 # Configuration management
├── utils/                  # Common utilities
└── analysis_template/      # Main package
```

## Core Components

### API Service (`api/`)

**Production-ready FastAPI service with auto-generated docs.**

```python
# Example: Custom endpoint
from src.api.routes import router
from src.api.schemas import PredictionRequest

@router.post("/custom-predict")
async def custom_predict(request: PredictionRequest):
    result = model_service.predict(request.features)
    return {"prediction": result}
```

**Key files:**
- `app.py` - Application factory and middleware
- `routes.py` - API endpoints
- `service.py` - Model management service
- `models.py` - Request/response schemas

### Data Processing (`data/`)

**Validation, transformation, and feature engineering.**

```python
# Example: Data pipeline
from src.data import load_data, process_features, validate_quality

df = load_data("data/raw/dataset.csv")
features = process_features(df, config={"scaling": "standard"})
quality_report = validate_quality(features)
```

**Key files:**
- `loading.py` - Data ingestion utilities
- `processing.py` - Feature engineering
- `validation.py` - Data quality checks

### Model Management (`models/`)

**Training, evaluation, and inference workflows.**

```python
# Example: Model training
from src.ml import train_model, evaluate_model

model = train_model(X_train, y_train, model_type="random_forest")
metrics = evaluate_model(model, X_test, y_test)
```

**Key files:**
- `training.py` - Training workflows
- `evaluation.py` - Model evaluation
- `inference.py` - Prediction interface

### Plugin System (`plugins/`)

**Extensible architecture for custom components.**

```python
# Example: Custom plugin
from src.plugins.base import BasePlugin
from src.plugins import register_plugin

@register_plugin("custom_processor", "data_processing")
class CustomProcessor(BasePlugin):
    def execute(self, context):
        # Custom logic
        return result
```

**Key files:**
- `base.py` - Abstract base classes
- `registry.py` - Plugin registration
- `__init__.py` - Plugin imports

### Configuration (`config/`)

**Type-safe configuration management with Hydra.**

```python
# Example: Using configs
from src.config import load_config

config = load_config()
model_params = config.model
api_settings = config.api
```

**Key files:**
- `manager.py` - Config loading and validation
- `models.py` - Pydantic schemas

## Development Patterns

### Adding New Components

1. **Data processors** → `src/data/`
2. **Model types** → `src/models/`
3. **API endpoints** → `src/api/routes.py`
4. **Plugins** → `src/plugins/`

### Configuration Override

```python
# Environment variables
export MODEL__TYPE=xgboost

# Command line
python script.py model.type=xgboost

# Config files
# conf/local.yaml
model:
  type: xgboost
```

### Testing

```bash
# Unit tests for specific modules
pytest tests/test_data.py -v
pytest tests/test_models.py -v
pytest tests/test_api.py -v

# Integration tests
pytest tests/integration/ -v
```

### Plugin Development

```python
# 1. Create plugin class
class MyPlugin(BasePlugin):
    def execute(self, context):
        return processed_data

# 2. Register plugin
register_plugin("my_plugin", "category")(MyPlugin)

# 3. Use in workflows
plugin = get_plugin("my_plugin")
result = plugin.execute(context)
```

## API Development

### Adding Endpoints

```python
# src/api/routes.py
@router.post("/api/v1/custom")
async def custom_endpoint(request: CustomRequest):
    # Processing logic
    return CustomResponse(result=data)
```

### Request/Response Models

```python
# src/api/models.py
class CustomRequest(BaseModel):
    data: List[float]
    options: Dict[str, Any]

class CustomResponse(BaseModel):
    result: Any
    status: str
```

### Service Integration

```python
# src/api/service.py
class ModelService:
    def predict(self, features, model_name="default"):
        model = self.models[model_name]
        return model.predict(features)
```

## Configuration Examples

### Model Configuration
```yaml
# conf/model/default.yaml
model:
  type: "random_forest"
  n_estimators: 100
  max_depth: 10
  random_state: 42
```

### API Configuration
```yaml
# conf/api/default.yaml
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  cors_origins: ["*"]
```

### Data Configuration
```yaml
# conf/data/default.yaml
data:
  source: "csv"
  path: "data/raw/"
  validation:
    required_columns: ["feature1", "feature2"]
    max_missing_ratio: 0.1
```

## Error Handling

### Data Validation
```python
from src.data.validation import ValidationError

try:
    df = validate_data(raw_data)
except ValidationError as e:
    logger.error(f"Data validation failed: {e}")
```

### Model Errors
```python
from src.ml.exceptions import ModelError

try:
    model = load_model("model.pkl")
except ModelError as e:
    logger.error(f"Model loading failed: {e}")
```

### API Errors
```python
from fastapi import HTTPException

@router.post("/predict")
async def predict(request: PredictionRequest):
    try:
        return model.predict(request.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Performance Tips

1. **Use async/await** for I/O operations
2. **Cache models** in memory for faster predictions
3. **Batch predictions** when possible
4. **Use connection pooling** for databases
5. **Monitor memory usage** for large datasets

## Security Considerations

1. **Validate all inputs** using Pydantic models
2. **Sanitize file paths** to prevent directory traversal
3. **Use environment variables** for secrets
4. **Enable CORS** only for trusted origins
5. **Log security events** for monitoring
