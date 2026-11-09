-- Disposable local-development fixture: the pytest database beside the dev database.
-- Extensions are created by the test fixture itself (the container user is a superuser).
CREATE DATABASE bddk_test;
