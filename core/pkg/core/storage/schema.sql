CREATE TABLE detection_status (
    request_id VARCHAR(255) PRIMARY KEY,
    status VARCHAR(255) NOT NULL,
    data BYTEA
);
