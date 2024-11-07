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
### Docker compose setup
Execute the following command from the project directory
```
 docker-compose up --build -d
```
1) The front-end is accessible from http://localhost:7860/
2) The back-end is accessible from http://core:8080

### Docker setup
- Frontend
    1. Build docker image (Your PWD has to be in the `frontend` directory)
    ```bash
     docker build -t front-end-mango-truth .
    ```
    2. Run docker container
    ```bash
     docker run -d --name front-end-mango-truth -p 7860:7860 front-end-mango-truth 
    ```
    3. Access the Gradio front-end using http://127.0.0.1:7860/ or http://localhost:7860/
- Core (bash)
    1. Build docker image (Your PWD has to be in the **project root** directory)
    ```bash
     docker build --tag mango-truth-core:1.0.0 --file ./core/Dockerfile .
    ```
    2. Run docker container
    ```bash
     docker run -d --name core -p 8080:8080 -e COMPUTE_HOST='host.docker.internal' mango-truth-core:1.0.0
    ```
    3. Access core using http://127.0.0.1:8080/ or http://localhost:8080/
- Core (With makefile)
  - Follow `MAKEFILE` commands to run core+rabbitmq in separate docker network. 
    - `network`
    - `run-rabbitmq`
    - `run-core`
  - Access core using http://127.0.0.1:8080/ or http://localhost:8080/

## Documentation

API endpoints are available in `core/docs/swagger.json` file.

To generate docs for core endpoints run the following command:

```bash 
 swag init -g ./core/cmd/server/main.go --output ./core/docs
```
