CREATE TABLE detection_status (
    request_id UUID PRIMARY KEY,
    status VARCHAR(255) NOT NULL,
    data BYTEA
);
