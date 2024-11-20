import logging
import traceback
import importlib
from compute.models.detectors import DetectorSignature
from detectors.interfaces import IDetector
from detectors.interfaces import Nexus
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor


def get_class_constructor(classpath: str):
    module_path, class_name = classpath.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class IDetectorsProvider:
    def get_detectors(self) -> List[DetectorSignature]:
        pass


class PostgresDetectorsProvider(IDetectorsProvider):
    def __init__(self, postgres_host: str, postgres_db: str, postgres_user: str, postgres_password: str):
        self.conn = psycopg2.connect(
            host=postgres_host,
            database=postgres_db,
            user=postgres_user,
            password=postgres_password,
            port=5432
        )

    def get_detectors(self) -> List[DetectorSignature]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT run_id, name, classpath FROM detectors")
            rows = cursor.fetchall()
            detectors = [
                DetectorSignature(
                    run_id=row['run_id'],
                    name=row['name'],
                    classpath=row['classpath']
                ) for row in rows
            ]
        return detectors


class ListDetectorsProvider(IDetectorsProvider):
    def __init__(self, detectors: List[DetectorSignature]):
        self.detectors = detectors

    def get_detectors(self) -> List[DetectorSignature]:
        return self.detectors


class MockDetectorsEngine:
    def __init__(self, detector: IDetector):
        self.detector = detector

    def get_detector_by_name(self, name: str) -> Optional[IDetector]:
        if name == 'mock_detector':
            return self.detector
        else:
            return None


class DetectorsEngine:
    def __init__(self, provider: IDetectorsProvider, nexus: Nexus):
        self.detectors_signatures = provider.get_detectors()
        self.detectors = {}
        for signature in self.detectors_signatures:
            try:
                detector: IDetector = get_class_constructor(signature.classpath)()
                detector.load_weights(nexus.load_run_weights(signature.run_id))
                self.detectors[signature.name] = detector
                logging.info(f"Successfully loaded detector {signature.name}.")
            except:
                logging.warning(f"Could not instantiate detector {signature.name}. Ex = " + traceback.format_exc())

    def get_detector_by_name(self, name: str) -> Optional[IDetector]:
        return self.detectors.get(name, None)
