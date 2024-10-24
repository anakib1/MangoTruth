# MangoTruth

[![CI Pipeline](https://github.com/anakib1/MangoTruth/actions/workflows/ci.yml/badge.svg)](https://github.com/anakib1/MangoTruth/actions/workflows/ci.yml)

## Purpose:

**MangoTruth** is an open srouce AI detection system focused on:

- Ukrainian language
- Academic domain
- Explainable models

## Ecosystem

Repository consists of:

- `frontend(py)` - user interface, authentication and requests routing
- `core(go)` - main application server that provides rest endpoints for both external integrations and `frontend`.
  Handles subscriptions, persistence, risk management, external integration and routes requests to execution venues
- `compute(py)` - compute engine of the system. Routes compute requests to detectors on both centralized and community
  servers
- `detectors(py)` - purely python code with implementation of known algorithms for AI detection implementing simple
  interface.

### Supported models:

| Name | Paper                                    | Benchmark score (*) |
|------|------------------------------------------|---------------------| 
| GLTR | [link](https://arxiv.org/pdf/1906.04043) | TBD                 |

## Configuration

To set up installation on your environment follow these steps:

- TBD
- TBD

## Documentation

API endpoints are available in `core/docs/swagger.json` file.

To generate docs for core endpoints run the following command:

```bash 
 swag init --dir core/main/ --output core/docs
```