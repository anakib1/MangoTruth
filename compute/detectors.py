import logging
import traceback
import importlib
from compute.models.detectors import Detector
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
    def get_detectors(self) -> List[Detector]:
        pass


class PostgresDetectorsProvider(IDetectorsProvider):
    def __init__(self, postgre_db: str, postgre_user: str, postgre_password: str):
        self.conn = psycopg2.connect(
            database=postgre_db,
            user=postgre_user,
            password=postgre_password
        )

    def get_detectors(self) -> List[Detector]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT run_id, name, classpath FROM detectors")
            rows = cursor.fetchall()
            detectors = [
                Detector(
                    run_id=row['run_id'],
                    name=row['name'],
                    classpath=row['classpath']
                ) for row in rows
            ]
        return detectors


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

    def get_detector(self, name: str) -> Optional[IDetector]:
        return self.detectors.get(name, None)
