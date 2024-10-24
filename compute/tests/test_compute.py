from uuid import uuid4
from compute.models.communication import ComputeRequest


class TestCompute:
    def test_compute_exists(self):
        req = ComputeRequest(uuid4(), content="Something", detector_name="GLTK")
