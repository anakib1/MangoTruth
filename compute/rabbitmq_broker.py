import json
import logging
import threading
import traceback
from dataclasses import asdict

import pika

from compute.interfaces import IMessageBroker
from compute.models.communication import ComputeRequest


class RabbitMQBroker(IMessageBroker):
    def __init__(self, source_queue_name, response_queue_name, rabbitmq_host: str, rabbitmq_port: int,
                 rabbitmq_username: str, rabbitmq_password: str):

        credentials = pika.PlainCredentials(username=rabbitmq_username, password=rabbitmq_password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=rabbitmq_host, port=rabbitmq_port, credentials=credentials))
        self.channel = self.connection.channel()

        self.source_queue_name = source_queue_name  # input queue
        self.response_queue_name = response_queue_name  # output queue

        self.channel.queue_declare(queue=self.source_queue_name, durable=True)
        self.channel.queue_declare(queue=self.response_queue_name, durable=True)
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
        # Start the RabbitMQ consuming loop
        try:
            self.channel.basic_consume(queue=self.source_queue_name, on_message_callback=self.on_request, auto_ack=True)
            self.logger.info("RabbitMQ started waiting for messages.")
            self.channel.start_consuming()
            self.is_consuming = True
        except Exception as e:
            logging.error(f"ERROR: {traceback.format_exc()}")
        finally:
            self.is_consuming = False

    def stop_consuming(self):
        if self.is_consuming:
            self.channel.stop_consuming()
            self.consumer_thread.join()  # Wait for the thread to finish
            self.is_consuming = False
            self.logger.info("RabbitMQ stopped waiting for messages.")

    def on_request(self, ch, method, properties, body):
        request = ComputeRequest.from_json(body.decode('utf-8'))

        response = self.process_request_method(request)
        serialized_response = json.dumps(asdict(response)).encode('utf-8')

        self.channel.basic_publish(
            exchange='',
            routing_key=self.response_queue_name,
            body=serialized_response,
            properties=pika.BasicProperties(delivery_mode=1)
        )
        self.logger.debug(f"Sent response for request ID: {request.request_id}")

    def close(self):
        self.connection.close()
        self.logger.info("RabbitMQ closed.")
