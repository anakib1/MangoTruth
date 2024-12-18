# MangoTruth

[![CI Pipeline](https://github.com/anakib1/MangoTruth/actions/workflows/ci.yml/badge.svg)](https://github.com/anakib1/MangoTruth/actions/workflows/ci.yml)

## Purpose:

**MangoTruth** is an open srouce AI detection system focused on:

- Ukrainian language
- Academic domain
- Explainable models

## Contribution

### New model

When adding new model, check out our [training guidelines](docs/training.md)!

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

For more info check [RESEARCH](RESEARCH.md)

| Name                | Paper                                    | val F1    | val AUC   | Neptune URL                                                                                                                                                                                             |
|---------------------|------------------------------------------|-----------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GLTR                | [link](https://arxiv.org/pdf/1906.04043) | TBD       | TBD       | TBD                                                                                                                                                                                                     |
| Ghostbuster         | [link](https://arxiv.org/pdf/2305.15047) | **0.981** | **0.971** | [link](https://app.neptune.ai/o/mango/org/mango-truth/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=Classification-report-9d8592d6-b5d9-47fa-ae63-75ee2b9e3439&shortId=MNGTRTH-23) |
| Perplexity baseline | -                                        | 0.662     | 0.5       | [link](https://app.neptune.ai/o/mango/org/mango-truth/runs/details?viewId=standard-view&detailsTab=dashboard&dashboardId=Classification-report-9d8592d6-b5d9-47fa-ae63-75ee2b9e3439&shortId=MNGTRTH-29) |

## Installation from source

To set up installation on your environment follow these steps:

- Core:
    - [Install go](https://go.dev/doc/install)
    - [Install rabbitMQ](https://www.rabbitmq.com/docs/download)
    - Configure server using `core/config/config-development.yml` file. Extensive configuration reference coming soon
    - [Install sqlboiler](https://github.com/volatiletech/sqlboiler)
    - Run `sqlboiler --wipe --output .\core\pkg\storage\models psql`
    - Run `go build ./core/cmd/server`
    - Run `go run core/cmd/server/main.go`
- Compute:
    - [Optional](https://docs.python.org/3/library/venv.html) create venv
    - pip install -r "compute/requirements.txt"
    - pip install -r "detectors/requirements.txt"
    - python "compute/main.py"
- Frontend:
    - [Install Python](https://www.python.org/downloads/release/python-3100/)
    - [Optional](https://docs.python.org/3/library/venv.html) create venv
    - pip install -r "frontend/requirements.txt"
    - python "frontend/app.py"

## Running as container

### Docker compose setup

Execute the following command from the project directory

```
 docker volume create storage
 docker-compose up --build -d
```

1) The front-end is accessible from http://localhost:7860/
2) The back-end is accessible from http://localhost:8080/ or http://core:8080/

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
        - `run-postgres`
    - Access core using http://127.0.0.1:8080/ or http://localhost:8080/
- Compute
  . Build docker image (Your PWD has to be in the **project root** directory)
    ```bash
     docker build --tag mango-truth-compute:1.0.0 --file ./compute/Dockerfile .
    ```
    2. Run docker container
    ```bash
     docker run -d --name compute -e RABBITMQ_HOST=host.docker.internal mango-truth-compute:1.0.0
    ```  

## Configuration

Generally, all variables mentioned here follow uppercase naming with "_" as delimiter for env variable and camelcase for
`.yml` config
files. Example:

Specifing `SERVER_PORT=8080` in env is equivalent to specifying in corresponding config location.

```yaml
server:
  port: 8080
```

### Core

Config is resolved by path `./core/config/config-development.yml`. TODO: Add option to override location.

| Variable               | Usage                                | Default    |
|------------------------|--------------------------------------|------------|
| `server.port`          | port of server on which core listens | 8080       |
| `compute.host`         | host of RabbitMQ                     | 127.0.0.1  |
| `compute.username`     | credentials for RabbitMQ             | guest      |
| `compute.password`     | credentials for RabbitMQ             | guest      |
| `storage.databaseName` | credentials for PostgreSQL           | mango-db   |
| `storage.userName`     | credentials for PostgreSQL           | mango-user |
| `storage.password`     | credentials for PostgreSQL           | password   |
| `storage.hostName`     | credentials for PostgreSQL           | postgres   |

## Frontend

| Variable             | Usage                                      | Default               |
|----------------------|--------------------------------------------|-----------------------|
| `server.url`         | url of server on which **compute** listens | http://localhost:8080 |
| `gradio.server.name` | URL on which frontend listens              | 127.0.0.1             |
| `gradio.server.port` | port on which frontend listens             | 7860                  |

## Compute

| Variable            | Usage                                | Default   |
|---------------------|--------------------------------------|-----------|
| `rabbitmq.port`     | port of server on which core listens | 8080      |
| `rabbitmq.host`     | host of RabbitMQ                     | 127.0.0.1 |
| `rabbitmq.username` | credentials for RabbitMQ             | guest     |
| `rabbitmq.password` | credentials for RabbitMQ             | guest     |

## Documentation

API endpoints are available in `core/docs/swagger.json` file.

To generate docs for core endpoints run the following command:

```bash 
cd core 
swag init -g ./cmd/server/main.go --output ./docs
```
