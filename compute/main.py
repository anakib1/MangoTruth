import os
import yaml
from compute.engine import ComputeEngine
from compute.rabbitmq_broker import RabbitMQBroker
from detectors.mocks import MockDetector


def load_config(file_path: str) -> dict:
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    try:
        # Load configuration
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config", "compute_config.yaml")
        config = load_config(config_path)

        # RabbitMQ Broker Configuration
        broker_config = config["rabbitmq"]
        broker = RabbitMQBroker(
            source_queue_name=broker_config["source_queue_name"],
            response_queue_name=broker_config["response_queue_name"],
            rabbitmq_host=broker_config["host"],
            rabbitmq_port=broker_config["port"]
        )

        # Mock Initialization of Detectors
        detectors = [MockDetector(detector["name"]) for detector in config["detectors"]]
        default_detector = detectors[0]

        # Engine init
        engine = ComputeEngine(detectors=detectors, default_detector=default_detector, broker=broker)

        print("Starting the Compute Engine...")
        engine.start_consuming()

        # Keep the program running
        while True:
            pass
    except KeyboardInterrupt:
        print("Stopping the Compute Engine...")
        engine.stop_consuming()
        engine.close()

