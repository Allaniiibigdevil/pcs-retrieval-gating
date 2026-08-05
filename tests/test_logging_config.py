import logging

from app.logging_config import CONSOLE_HANDLER_NAME, configure_logging


def test_configure_logging_installs_one_application_console_handler() -> None:
    app_logger = configure_logging()
    configure_logging()

    console_handlers = [
        handler
        for handler in app_logger.handlers
        if handler.get_name() == CONSOLE_HANDLER_NAME
    ]

    assert app_logger.level == logging.INFO
    assert app_logger.propagate is False
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.INFO
