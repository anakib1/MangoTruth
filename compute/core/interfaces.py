class IMessageBroker:

    def set_process_request_method(self, process_request_method) -> None:
        """FUNCTION THAT SETS INCOMING REQUEST DATA PRECESSING FUNCTION"""
        pass

    def start_consuming(self) -> None:
        """FUNCTION THAT STARTS CONSUMING CYCLE ON BACKGROUND"""
        pass

    def stop_consuming(self) -> None:
        """FUNCTION THAT STOPS CONSUMING CYCLE"""
        pass

    def on_request(self, ch, method, properties, body) -> None:
        """FUNCTION THAT DEFINES BEHAVIOR ON UPON RECIEVING REQUEST """
        pass

    def close(self):
        """CLOSES ENGINE"""
        pass