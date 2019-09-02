DROP TABLE IF EXISTS ClientAuthInfo;

CREATE TABLE ClientAuthInfo
(
    client_name TEXT,
    username TEXT,
    machine_name TEXT,
    PRIMARY KEY(client_name, username, machine_name)
);