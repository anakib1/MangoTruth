import json
import queue
import threading

from compute.interfaces import IMessageBroker
from compute.models.communication import ComputeRequest


class LocalMessageBroker(IMessageBroker):
    def __init__(self):
        self.source_queue = queue.Queue()  # Queue for incoming messages
        self.response_queue = queue.Queue()  # Queue for responses

        self.process_request_method = None
        self.consumer_thread = None  # Thread for consuming messages
        self.is_consuming = False  # Flag to manage the consumer lifecycle

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
        print("LocalMessageBroker started waiting for messages.")
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
            print("LocalMessageBroker stopped waiting for messages.")

    def on_request(self, ch, method, properties, body):
        request_data = json.loads(body.decode('utf-8'))
        request = ComputeRequest(
            request_id=request_data['request_id'],
            content=request_data['content'],
            detector_name=request_data['detector_name']
        )

        response = self.process_request_method(request)
        response_dict = {
            'explanation': response.explanation,
            'predictions': response.predictions,
            'request_id': response.request_id
        }
        serialized_response = json.dumps(response_dict).encode('utf-8')

        # Instead of sending through a network, put the response in the response queue
        self.response_queue.put(serialized_response)
        print(f" [x] Sent response for request ID: {request.request_id}")

    def close(self):
        self.stop_consuming()
        print("LocalMessageBroker closed.")

    # For testing purposes: Method to add a request to the queue
    def publish_request(self, request_data):
        self.source_queue.put(json.dumps(request_data).encode('utf-8'))

    # For testing purposes: Method to retrieve a response from the queue
    def get_response(self):
        try:
            return self.response_queue.get_nowait().decode('utf-8')
        except queue.Empty:
            return None
