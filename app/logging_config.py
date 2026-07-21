import logging


APP_LOGGER_NAME = "app"
CONSOLE_HANDLER_NAME = "pcs-console"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure application logs without replacing Uvicorn's handlers."""

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level)
    app_logger.propagate = False

    console_handler = next(
        (
            handler
            for handler in app_logger.handlers
            if handler.get_name() == CONSOLE_HANDLER_NAME
        ),
        None,
    )
    if console_handler is None:
        console_handler = logging.StreamHandler()
        console_handler.set_name(CONSOLE_HANDLER_NAME)
        app_logger.addHandler(console_handler)

    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return app_logger
