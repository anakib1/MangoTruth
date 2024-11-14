import json
import logging
import queue
import threading
from dataclasses import asdict

from compute.interfaces import IMessageBroker
from compute.models.communication import ComputeRequest


class MockMessageBroker(IMessageBroker):
    def __init__(self):
        self.source_queue = queue.Queue()  # Queue for incoming messages
        self.response_queue = queue.Queue()  # Queue for responses

        self.process_request_method = None
        self.consumer_thread = None  # Thread for consuming messages
        self.is_consuming = False  # Flag to manage the consumer lifecycle
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logger = logging.getLogger(__name__)

    def set_process_request_method(self, process_request_method):
        self.process_request_method = process_request_method

    def start_consuming(self):
        if not self.is_consuming:
            # Start the consumer thread
            self.consumer_thread = threading.Thread(target=self._consume)
            self.consumer_thread.daemon = True
            self.consumer_thread.start()

    def _consume(self):
        # Start the consuming loop for the local queue
        self.is_consuming = True
        self.logger.info("MOCKRabbitMQ started waiting for messages.")
        while self.is_consuming:
            try:
                body = self.source_queue.get(timeout=1)  # Get message with a timeout
                self.on_request(None, None, None, body)
            except queue.Empty:
                continue  # Continue looping if no message is available

    def stop_consuming(self):
        if self.is_consuming:
            self.is_consuming = False
            self.consumer_thread.join()  # Wait for the thread to finish
            self.logger.info("MOCKRabbitMQ stopped waiting for messages.")

    def on_request(self, ch, method, properties, body):
        request = ComputeRequest.from_json(body.decode('utf-8'))

        response = self.process_request_method(request)
        serialized_response = json.dumps(asdict(response)).encode('utf-8')

        # Instead of sending through a network, put the response in the response queue
        self.response_queue.put(serialized_response)
        self.logger.debug(f"Sent response for request ID: {request.request_id}")

    def close(self):
        self.stop_consuming()
        self.logger.info("MOCKRabbitMQ closed.")

    # For testing purposes: Method to add a request to the queue
    def publish_request(self, request_data):
        self.source_queue.put(json.dumps(request_data).encode('utf-8'))

    # For testing purposes: Method to retrieve a response from the queue
    def get_response(self):
        try:
            return self.response_queue.get_nowait().decode('utf-8')
        except queue.Empty:
            return None
