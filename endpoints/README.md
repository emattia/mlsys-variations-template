# Production API Endpoints

This directory contains production-ready API endpoints for serving ML models. The API is built with FastAPI and provides high-performance, scalable model serving with automatic validation, documentation, and monitoring.

## 🎯 **Purpose**

The endpoints directory provides **ML platform operators and data scientists** with:

- **🚀 Production-Ready API**: FastAPI-based model serving with automatic validation
- **📊 Model Management**: Load, manage, and serve multiple ML models
- **🔒 Security Features**: Authentication, rate limiting, and input validation
- **📈 Monitoring**: Built-in metrics collection and health checks
- **📝 Auto-Documentation**: Interactive OpenAPI documentation
- **🐳 Container Ready**: Docker deployment with optimized performance

## 🏗️ **API Architecture**

```
endpoints/
├── 🚀 Production API (src/api/)
│   ├── app.py              # FastAPI application factory
│   ├── routes.py           # API endpoint definitions
│   ├── service.py          # Model management service
│   ├── models.py           # Pydantic validation models
│   └── middleware.py       # Security & monitoring middleware
│
├── 📊 Model Management
│   ├── health/             # Health check endpoints
│   ├── metrics/            # Prometheus metrics
│   └── admin/              # Administrative endpoints
│
├── 🧪 Development & Testing
│   ├── test_endpoints.py   # API integration tests
│   └── examples/           # Usage examples
│
└── 📋 Configuration
    ├── api_config.yaml     # API configuration
    └── security.yaml       # Security settings
```

---

## 🚀 **Quick Start**

### **Start Development Server**

```bash
# Start development server with hot reload
make serve-dev

# API will be available at:
# - API Documentation: http://localhost:8000/docs
# - Health Check: http://localhost:8000/api/v1/health
# - Metrics: http://localhost:8000/api/v1/metrics
```

### **Make Your First Prediction**

```bash
# Test the prediction endpoint
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [5.1, 3.5, 1.4, 0.2],
       "model_name": "default"
     }'

# Response:
# {
#   "prediction": 0,
#   "prediction_label": "setosa",
#   "probabilities": [0.95, 0.03, 0.02],
#   "model_name": "default",
#   "prediction_time": 0.003,
#   "timestamp": "2024-12-07T10:30:00Z"
# }
```

### **Load Custom Models**

```bash
# Load a new model via API
curl -X POST "http://localhost:8000/api/v1/models/load" \
     -H "Content-Type: application/json" \
     -d '{
       "model_name": "custom_model",
       "model_path": "models/trained/custom_model.pkl"
     }'
```

---

## 🔧 **API Endpoints**

### **Model Prediction Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Single prediction with default model |
| `/api/v1/predict/{model_name}` | POST | Single prediction with specific model |
| `/api/v1/batch-predict` | POST | Batch predictions (up to 1000 samples) |
| `/api/v1/predict-async` | POST | Async prediction for large datasets |

### **Model Management Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/models` | GET | List all loaded models |
| `/api/v1/models/load` | POST | Load a new model |
| `/api/v1/models/{model_name}` | DELETE | Unload a model |
| `/api/v1/models/{model_name}/info` | GET | Get model metadata |

### **System Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | System health check |
| `/api/v1/metrics` | GET | Prometheus metrics |
| `/api/v1/status` | GET | Detailed system status |
| `/docs` | GET | Interactive API documentation |

---

## 💡 **Usage Examples**

### **Single Prediction**

```python
import requests

# Basic prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "features": [5.1, 3.5, 1.4, 0.2],
        "return_probabilities": True
    }
)

prediction = response.json()
print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {max(prediction['probabilities']):.3f}")
```

### **Batch Predictions**

```python
import requests

# Batch prediction for multiple samples
response = requests.post(
    "http://localhost:8000/api/v1/batch-predict",
    json={
        "features": [
            [5.1, 3.5, 1.4, 0.2],
            [6.3, 2.9, 5.6, 1.8],
            [4.9, 3.0, 1.4, 0.2]
        ],
        "model_name": "iris_classifier"
    }
)

results = response.json()
for i, pred in enumerate(results['predictions']):
    print(f"Sample {i+1}: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
```

### **Async Predictions for Large Datasets**

```python
import requests
import time

# Submit async prediction job
response = requests.post(
    "http://localhost:8000/api/v1/predict-async",
    json={
        "data_path": "data/batch/large_dataset.csv",
        "output_path": "predictions/results.csv",
        "model_name": "production_model"
    }
)

job_id = response.json()['job_id']

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8000/api/v1/jobs/{job_id}/status")
    if status.json()['status'] == 'completed':
        break
    time.sleep(5)

print("Predictions complete!")
```

### **Model Management**

```python
import requests

# List available models
models = requests.get("http://localhost:8000/api/v1/models").json()
print(f"Available models: {[m['name'] for m in models['models']]}")

# Load a new model
response = requests.post(
    "http://localhost:8000/api/v1/models/load",
    json={
        "model_name": "new_model",
        "model_path": "models/trained/new_model.pkl",
        "model_type": "sklearn"
    }
)

if response.status_code == 200:
    print("Model loaded successfully!")

# Get model information
info = requests.get("http://localhost:8000/api/v1/models/new_model/info").json()
print(f"Model info: {info}")
```

---

## 🔒 **Security Features**

### **Input Validation**

All endpoints use Pydantic models for automatic input validation:

```python
# Request validation model
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_items=1, max_items=1000)
    model_name: str = Field(default="default", regex="^[a-zA-Z0-9_-]+$")
    return_probabilities: bool = Field(default=False)

    @validator('features')
    def validate_features(cls, v):
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError('All features must be numeric')
        return v
```

### **Rate Limiting**

Built-in rate limiting to prevent abuse:

```python
# Rate limiting configuration
RATE_LIMITS = {
    "predict": "100/minute",
    "batch_predict": "10/minute",
    "model_management": "20/minute"
}
```

### **Authentication**

Support for multiple authentication methods:

```bash
# API key authentication
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"features": [1, 2, 3, 4]}'

# JWT authentication
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Authorization: Bearer your-jwt-token" \
     -H "Content-Type: application/json" \
     -d '{"features": [1, 2, 3, 4]}'
```

---

## 📊 **Monitoring & Observability**

### **Built-in Metrics**

The API automatically collects metrics for monitoring:

```python
# Available metrics at /api/v1/metrics
http_requests_total{method="POST", endpoint="/api/v1/predict", status="200"}
prediction_duration_seconds{model_name="default"}
model_predictions_total{model_name="default", status="success"}
active_models_gauge
memory_usage_bytes
cpu_usage_percent
```

### **Health Checks**

Comprehensive health monitoring:

```python
# Health check response
{
    "status": "healthy",
    "timestamp": "2024-12-07T10:30:00Z",
    "checks": {
        "database": "healthy",
        "models": {
            "loaded": 3,
            "status": "healthy"
        },
        "memory": {
            "used_percent": 45.2,
            "status": "healthy"
        },
        "disk": {
            "used_percent": 23.1,
            "status": "healthy"
        }
    },
    "version": "1.0.0"
}
```

### **Structured Logging**

JSON-formatted logs for easy analysis:

```json
{
    "timestamp": "2024-12-07T10:30:00Z",
    "level": "INFO",
    "message": "Prediction completed",
    "model_name": "default",
    "prediction_time": 0.003,
    "features_count": 4,
    "request_id": "req-123",
    "user_id": "user-456"
}
```

---

## 🐳 **Production Deployment**

### **Docker Deployment**

The API is containerized for easy deployment:

```bash
# Build production container
make docker-build

# Run in production mode
make docker-prod

# Container will expose port 8000 with:
# - Optimized Python runtime
# - Non-root user for security
# - Health check endpoints
# - Resource limits
```

### **Docker Compose with Monitoring**

```yaml
# docker-compose.yml - Production stack
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### **Kubernetes Deployment**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: api
        image: ml-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## 🧪 **Testing**

### **API Testing**

Comprehensive test suite for API endpoints:

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint(client):
    """Test prediction endpoint with valid data."""
    response = client.post(
        "/api/v1/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "prediction_time" in data

def test_batch_prediction(client):
    """Test batch prediction endpoint."""
    response = client.post(
        "/api/v1/batch-predict",
        json={
            "features": [
                [5.1, 3.5, 1.4, 0.2],
                [6.3, 2.9, 5.6, 1.8]
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2

def test_invalid_input(client):
    """Test API with invalid input."""
    response = client.post(
        "/api/v1/predict",
        json={"features": "invalid"}
    )
    assert response.status_code == 422  # Validation error
```

### **Load Testing**

Performance testing with various tools:

```bash
# Using Apache Bench
ab -n 1000 -c 10 -T application/json -p test_data.json \
   http://localhost:8000/api/v1/predict

# Using wrk
wrk -t12 -c400 -d30s --script=test_script.lua \
   http://localhost:8000/api/v1/predict

# Using locust
locust -f load_test.py --host=http://localhost:8000
```

---

## ⚙️ **Configuration**

### **API Configuration**

```yaml
# conf/api_config.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30
  max_request_size: 10485760  # 10MB

security:
  enable_cors: true
  cors_origins: ["http://localhost:3000"]
  api_key_header: "X-API-Key"
  rate_limit_enabled: true

models:
  auto_load: true
  model_directory: "models/trained"
  max_models: 5
  model_timeout: 300

logging:
  level: "INFO"
  format: "json"
  access_log: true

monitoring:
  enable_metrics: true
  metrics_endpoint: "/api/v1/metrics"
  health_endpoint: "/api/v1/health"
```

### **Environment Variables**

```bash
# .env file
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO
ENVIRONMENT=production
MODEL_DIRECTORY=models/trained
ENABLE_METRICS=true
CORS_ENABLED=true
```

---

## 📚 **Best Practices**

### **API Development**

1. **📝 Document everything**: Use comprehensive docstrings and OpenAPI specs
2. **✅ Validate inputs**: Use Pydantic models for all request/response validation
3. **🛡️ Handle errors gracefully**: Provide meaningful error messages and status codes
4. **📊 Monitor performance**: Add metrics for all critical operations
5. **🔒 Secure endpoints**: Implement authentication and rate limiting
6. **🧪 Test thoroughly**: Write unit, integration, and load tests
7. **📦 Version APIs**: Include version information in URL paths

### **Model Management**

1. **♻️ Lazy loading**: Load models only when needed to save memory
2. **🔄 Graceful updates**: Support hot model swapping without downtime
3. **💾 Memory management**: Monitor and limit memory usage per model
4. **📈 Performance tracking**: Monitor prediction latency and throughput
5. **🚨 Error handling**: Gracefully handle model loading and prediction failures

### **Production Deployment**

1. **🐳 Use containers**: Deploy with Docker for consistency
2. **⚖️ Load balancing**: Use multiple replicas behind a load balancer
3. **📊 Monitor everything**: Set up comprehensive monitoring and alerting
4. **🔄 Blue-green deployment**: Use zero-downtime deployment strategies
5. **📦 Resource limits**: Set appropriate CPU/memory limits
6. **🔒 Security hardening**: Use non-root users and minimal base images

---

## 🤝 **Contributing**

1. **Add new endpoints** in `src/api/routes.py`
2. **Create Pydantic models** in `src/api/models.py`
3. **Write comprehensive tests** in `tests/test_api.py`
4. **Update API documentation** in docstrings
5. **Test with real models** before submitting PRs
6. **Follow REST conventions** for endpoint design

---

## 📚 **Additional Resources**

- **[FastAPI Documentation](https://fastapi.tiangolo.com/)** - FastAPI framework guide
- **[Pydantic Documentation](https://pydantic-docs.helpmanual.io/)** - Data validation
- **[Docker Documentation](../Dockerfile)** - Container deployment
- **[Monitoring Guide](../docs/monitoring.md)** - Production monitoring
- **[Security Guide](../docs/security.md)** - API security best practices
