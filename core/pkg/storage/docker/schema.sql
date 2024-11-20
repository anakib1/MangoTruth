CREATE TYPE STATUS AS ENUM ('UNKNOWN', 'PENDING', 'SUCCESS', 'FAILURE');

CREATE TABLE detection_status
(
    request_id UUID PRIMARY KEY,
    status     STATUS NOT NULL,
    data       BYTEA
);

CREATE TABLE detectors
(
    name      VARCHAR PRIMARY KEY,
    classpath VARCHAR,
    run_id    UUID
);

INSERT INTO detectors (name, classpath, run_id)
VALUES ('ghostbuster', 'detectors.ghostbuster.model.GhostbusterDetector',
        '85a90366-991b-4747-b247-86f146265318'),
       ('perplexity', 'detectors.perplexity.model.PerplexityModel', 'f2472ed5-936e-4b45-b3a0-d0dda3507f76');

COMMIT;

