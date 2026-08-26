-- LOCAL DEVELOPMENT ONLY.
--
-- Approved extensions for the disposable `bddk_test` pytest database. Runs
-- against bddk_test (see the bddk-test-db Compose service), matching what CI
-- provides inside its postgres service container.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public;
