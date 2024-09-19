# src/utils.py

import logging

def setup_logging(log_file: str = "training.log"):
    """
    로깅 설정을 초기화합니다.

    Args:
        log_file (str, optional): 로그 파일 경로. Defaults to "training.log".
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )
