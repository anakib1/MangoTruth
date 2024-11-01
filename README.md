# MangoTruth

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

## Installation from source

To set up installation on your environment follow these steps:

- Core:
    - [Install go](https://go.dev/doc/install)
    - [Install rabbitMQ](https://www.rabbitmq.com/docs/download)
    - Configure server using `core/config/config-development.yml` file. Extensive configuration reference coming soon
    - Run `go build ./core/cmd/server`
    - Run `go run core/cmd/server/main.go`
- Compute:
    - TBD
- Frontend:
    - [Install Python](https://www.python.org/downloads/release/python-3100/)
    - [Optional](https://docs.python.org/3/library/venv.html) create venv
    - pip install -r "frontend/requirements.txt"
    - python "frontend/app.py"

## Running as container

TBD

## Documentation

API endpoints are available in `core/docs/swagger.json` file.

To generate docs for core endpoints run the following command:

```bash 
 swag init -g ./core/cmd/server/main.go --output ./core/docs
```