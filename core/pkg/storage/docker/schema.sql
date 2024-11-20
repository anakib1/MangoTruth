CREATE TYPE STATUS AS ENUM('UNKNOWN', 'PENDING', 'SUCCESS', 'FAILURE');

CREATE TABLE detection_status (
    request_id UUID PRIMARY KEY,
    status STATUS NOT NULL,
    data BYTEA
);

CREATE TABLE detectors (
    name VARCHAR PRIMARY KEY,
    classpath VARCHAR,
    run_id UUID
);

INSERT INTO detectors (name, classpath, run_id)
VALUES
    ('ghostbuster', 'detectors.ghostbuster.model', 'df07a8d5-6cce-42c9-ab9f-6f5f1dca621f'),
    ('dna-gpt', 'detectors.dna.model', 'e7001002-6f18-4912-99f1-1c13ad532d39');

COMMIT;