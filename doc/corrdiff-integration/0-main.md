# CorrDiff Earth2Studio Integration Guide

> **🎯 Objective:** Build an Earth2Studio inference pipeline for your trained CorrDiff model within 24 hours

## 📋 Table of Contents

### [1. Quick Start](1-quickstart/README.md)
- [1.1 Prerequisites Check](1-quickstart/1.1-prerequisites.md)
- [1.2 Environment Setup](1-quickstart/1.2-environment.md) 
- [1.3 Validate Current Pipeline](1-quickstart/1.3-validate.md)

### [2. Understanding CorrDiff Architecture](2-architecture/README.md)
- [2.1 Earth2Studio CorrDiff Implementation](2-architecture/2.1-implementation.md)
- [2.2 Checkpoint Structure Analysis](2-architecture/2.2-checkpoints.md)
- [2.3 Model Loading Pipeline](2-architecture/2.3-loading.md)

### [3. Custom Model Integration](3-integration/README.md)
- [3.1 Checkpoint Format Conversion](3-integration/3.1-conversion.md)
- [3.2 Custom CorrDiff Class](3-integration/3.2-custom-class.md)
- [3.3 Coordinate System Mapping](3-integration/3.3-coordinates.md)

### [4. Data Pipeline Setup](4-data/README.md)
- [4.1 Input Data Preparation](4-data/4.1-input-prep.md)
- [4.2 Custom Data Source](4-data/4.2-datasource.md)
- [4.3 Validation & Testing](4-data/4.3-validation.md)

### [5. Inference Pipeline](5-inference/README.md)
- [5.1 Custom Workflow Creation](5-inference/5.1-workflow.md)
- [5.2 Running Inference](5-inference/5.2-execution.md)
- [5.3 Output Processing](5-inference/5.3-postprocess.md)

### [6. Testing & Validation](6-testing/README.md)
- [6.1 Unit Tests](6-testing/6.1-unit-tests.md)
- [6.2 Integration Tests](6-testing/6.2-integration.md)
- [6.3 Performance Benchmarks](6-testing/6.3-benchmarks.md)

### [7. Production Deployment](7-deployment/README.md)
- [7.1 Docker Configuration](7-deployment/7.1-docker.md)
- [7.2 Scaling Considerations](7-deployment/7.2-scaling.md)
- [7.3 Monitoring & Logging](7-deployment/7.3-monitoring.md)

---

## 🚀 Quick Navigation

| Phase | Time Estimate | Key Files |
|-------|---------------|-----------|
| **Setup & Validation** | 2-4 hours | `1-quickstart/` |
| **Architecture Understanding** | 3-5 hours | `2-architecture/` |
| **Model Integration** | 8-12 hours | `3-integration/`, `4-data/` |
| **Testing & Deployment** | 4-6 hours | `5-inference/`, `6-testing/` |

## 📁 Key Files Referenced

### Your Checkpoints
- `/home/younes.abid/git/physicsnemo/outputs/checkpoints/*/checkpoints_regression/*.mdlus`
- `/home/younes.abid/git/physicsnemo/outputs/checkpoints/*/checkpoints_diffusion/*.mdlus`

### Earth2Studio Reference
- `/home/younes.abid/git/earth2studio/examples/04_corrdiff_inference.py`
- `/home/younes.abid/git/earth2studio/earth2studio/models/dx/corrdiff.py`
- `/home/younes.abid/git/earth2studio/scripts/`

### Target Implementation
- `custom_corrdiff_model.py` *(to be created)*
- `custom_inference_workflow.py` *(to be created)*
- `validation_tests.py` *(to be created)*

## 🎪 Success Criteria

- [ ] Load your trained checkpoints successfully
- [ ] Run inference on your custom dataset
- [ ] Generate high-quality downscaled outputs
- [ ] Validate results match expected quality
- [ ] Deploy in Docker environment

---

*Last updated: January 26, 2026*