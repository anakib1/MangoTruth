import logging
import os
import traceback
from typing import Any

import yaml
from dotenv import load_dotenv

from compute.core.detectors import DetectorsEngine, PostgresDetectorsProvider
from compute.core.engine import ComputeEngine
from compute.core.rabbitmq_broker import RabbitMQBroker
from detectors.neptune.nexus import NeptuneNexus


def load_config(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def get_config_value(key: str, default: Any) -> Any:
    """Fetch a configuration value from the environment or fallback to a default."""
    return os.getenv(key, default)


if __name__ == "__main__":
    try:
        # Load configuration
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config", "compute_config.yaml")
        config = load_config(config_path)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(dotenv_path=os.path.join(base_dir, "config", "sample.env"))
        broker_config = config.get("rabbitmq", dict())
        detector_config = config.get("detectors", dict())

        detector_config["postgres_db"] = os.getenv("POSTGRES_DB", detector_config.get("postgres_db"))
        detector_config["postgres_user"] = os.getenv("POSTGRES_USER", detector_config.get("postgres_user"))
        detector_config["postgres_password"] = os.getenv("POSTGRES_PASSWORD", detector_config.get("postgres_password"))
        detector_config["postgres_host"] = os.getenv("POSTGRES_HOST", detector_config.get("postgres_host"))

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Using config {}'.format(config))

        # RabbitMQ Broker Configuration
        broker = RabbitMQBroker(
            source_queue_name=get_config_value("RABBITMQ_SOURCE_QUEUE_NAME", broker_config.get("source_queue_name")),
            response_queue_name=get_config_value("RABBITMQ_RESPONSE_QUEUE_NAME",
                                                 broker_config.get("response_queue_name")),
            rabbitmq_host=get_config_value("RABBITMQ_HOST", broker_config.get("host")),
            rabbitmq_port=int(get_config_value("RABBITMQ_PORT", broker_config.get("port"))),
            rabbitmq_username=get_config_value("RABBITMQ_USERNAME", broker_config.get("username")),
            rabbitmq_password=get_config_value("RABBITMQ_PASSWORD", broker_config.get("password"))
        )

        detection_provider = PostgresDetectorsProvider(
            detector_config['postgres_host'],
            detector_config['postgres_db'],
            detector_config['postgres_user'],
            detector_config['postgres_password'])

        try:
            nexus = NeptuneNexus()
        except Exception:
            logging.error("Failed to connect to neptune nexus. " + traceback.format_exc())
            nexus = None
        detector_engine = DetectorsEngine(detection_provider, nexus)

        engine = ComputeEngine(detectors_engine=detector_engine, broker=broker)
        logging.info("Starting the Compute Engine...")
        engine.start_consuming()

        # Keep the program running
        while True:
            pass
    except KeyboardInterrupt:
        logging.info("Stopping the Compute Engine...")
        engine.stop_consuming()
        engine.close()
