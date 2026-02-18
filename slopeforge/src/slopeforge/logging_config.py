import logging
import uuid


def generate_run_id() -> str:
    return uuid.uuid4().hex[:10]


def setup_logging(run_id: str) -> logging.LoggerAdapter:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s run_id=%(run_id)s %(message)s",
    )
    base_logger = logging.getLogger("slopeforge")
    return logging.LoggerAdapter(base_logger, {"run_id": run_id})


def get_logger(name: str, run_id: str) -> logging.LoggerAdapter:
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, {"run_id": run_id})
