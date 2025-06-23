# MLX Architecture: Component Injection Model

> **Like shadcn for ML Systems** - Components are injected into your source code, not external dependencies

## 🎯 **The Right Mental Model**

MLX works **exactly like shadcn/ui** but for ML components:

### **shadcn/ui Pattern:**
```bash
npx shadcn-ui@latest add button
# → Adds src/components/ui/button.tsx to your codebase
# → Code becomes part of YOUR project
# → Customizable, owned by you
```

### **MLX Pattern:**
```bash
mlx add api-serving
# → Injects FastAPI components into src/api/
# → Code becomes part of YOUR project
# → Customizable, owned by you
```

---

## 🏗️ **MLX Directory Structure**

### **✅ Correct MLX Model:**

```
your-project/
├── src/                           # ← Components get injected HERE
│   ├── api/                       # ← mlx add api-serving updates this
│   ├── config/                    # ← mlx add config-management updates this
│   ├── plugins/                   # ← mlx add plugin-registry updates this
│   └── utils/                     # ← mlx add caching updates this
│
├── mlx-components/                # ← Component registry/templates (like shadcn registry)
│   ├── registry.json             # ← Available components catalog
│   ├── api-serving/              # ← Templates for FastAPI components
│   ├── config-management/        # ← Templates for Hydra/Pydantic config
│   └── caching/                  # ← Templates for intelligent caching
│
└── scripts/mlx/                  # ← MLX CLI implementation
    ├── extract_component.py     # ← Extract from existing src/
    ├── inject_component.py      # ← Inject into src/
    └── cli.py                   # ← Main CLI commands
```

### **❌ Confused Model (What we had):**
```
your-project/
├── src/                     # ← Your actual source code
├── mlx/                     # ← ??? Unclear what this is
├── mlx-components/          # ← Component templates
└── scripts/mlx/             # ← MLX tooling
```

---

## 🚀 **How MLX Component Injection Works**

### **1. Component Registry (like shadcn)**
```json
// mlx-components/registry.json
{
  "api-serving": {
    "description": "FastAPI application with security & monitoring",
    "files": [
      "src/api/app.py",
      "src/api/middleware.py",
      "src/api/models.py"
    ],
    "dependencies": ["fastapi>=0.110.0", "uvicorn[standard]>=0.30.0"],
    "templates": ["mlx-components/api-serving/"]
  }
}
```

### **2. Component Injection Process**
```bash
# User runs command
mlx add api-serving

# MLX CLI does:
# 1. Read mlx-components/registry.json
# 2. Copy templates from mlx-components/api-serving/
# 3. Inject/merge into src/api/
# 4. Update dependencies in requirements.txt
# 5. Run compatibility checks
```

### **3. Intelligent Merging (Beyond shadcn)**
```python
# MLX handles smart merging when components overlap
mlx add api-serving      # Creates src/api/app.py
mlx add rate-limiting    # Enhances EXISTING src/api/app.py with rate limiting
mlx add caching          # Enhances EXISTING src/api/app.py with caching
```

---

## 🧩 **Component Types & Injection Targets**

| Component | Injects Into | Purpose |
|-----------|--------------|---------|
| `api-serving` | `src/api/` | FastAPI application framework |
| `config-management` | `src/config/` | Hydra + Pydantic configuration |
| `plugin-registry` | `src/plugins/` | Extensible plugin system |
| `caching` | `src/utils/cache/` | Intelligent LLM response caching |
| `rate-limiting` | `src/utils/rate/` | API rate limiting with cost tracking |
| `template-manager` | `src/utils/templates/` | Prompt template versioning |
| `llm-integration` | `src/llm/` | OpenAI/Anthropic integration |
| `vector-store` | `src/vector/` | Embeddings and similarity search |

---

## 🔧 **Implementation Strategy**

### **Phase 2: Component Extraction**
```bash
# Extract existing code into component templates
projen mlx:extract-components

# Creates templates in mlx-components/ from existing src/
src/api/          → mlx-components/api-serving/
src/config/       → mlx-components/config-management/
src/plugins/      → mlx-components/plugin-registry/
```

### **Phase 3: Component Injection CLI**
```bash
# Implement shadcn-style CLI
mlx add api-serving
mlx add caching
mlx remove rate-limiting
mlx list
mlx status
```

### **Phase 4: AI-Enhanced Intelligence**
```bash
# AI compatibility prediction
mlx add vector-store
# → AI: "⚠️  This conflicts with existing caching. Suggest: mlx add vector-store --merge-cache"

# Smart recommendations
mlx status
# → AI: "💡 Your API could benefit from rate limiting. Run: mlx add rate-limiting"
```

---

## 📁 **File Structure After Component Injection**

### **Before MLX:**
```
src/
├── api/app.py          # Basic FastAPI app
├── config/settings.py  # Simple config
└── utils/common.py     # Basic utilities
```

### **After `mlx add api-serving caching rate-limiting`:**
```
src/
├── api/
│   ├── app.py          # ← Enhanced with middleware
│   ├── middleware.py   # ← Added by api-serving
│   ├── security.py     # ← Added by api-serving
│   └── monitoring.py   # ← Added by api-serving
├── config/
│   ├── settings.py     # ← Enhanced by config-management
│   ├── models.py       # ← Added by config-management
│   └── manager.py      # ← Added by config-management
└── utils/
    ├── cache/          # ← Added by caching component
    │   ├── manager.py
    │   └── decorators.py
    └── rate/           # ← Added by rate-limiting component
        ├── limiter.py
        └── middleware.py
```

---

## 🎯 **Key Principles**

1. **Code Ownership**: Components become part of YOUR codebase (like shadcn)
2. **Customizable**: Modify injected code however you want
3. **Composable**: Components enhance each other intelligently
4. **AI-Powered**: Smart compatibility and recommendations
5. **Production-Ready**: All components are enterprise-grade

---

## 🚀 **Why This is Better Than Traditional Packages**

### **Traditional Python Packages:**
```python
# External dependency, hard to customize
from some_ml_package import APIFramework
app = APIFramework()  # Black box, limited customization
```

### **MLX Component Injection:**
```python
# YOUR code, injected by MLX, fully customizable
from src.api.app import create_app
from src.utils.cache import CacheManager

app = create_app()  # YOUR code, modify however you want
```

---

**Next Step**: Remove the confusing `mlx/` directory and implement the injection model properly in Phase 2.
