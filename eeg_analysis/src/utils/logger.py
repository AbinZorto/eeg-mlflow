import logging
import sys
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger
import datetime
import os

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""
    
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.datetime.utcnow().isoformat()
            
        # Add log level
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname
            
        # Add module and function information
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    json_format: bool = True
) -> logging.Logger:
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        console_level: Logging level for console output
        file_level: Logging level for file output
        json_format: Whether to use JSON formatting
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    if json_format:
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(module)s %(funcName)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily rotating log file
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d')
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(
    name: Optional[str] = None,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Get logger with default configuration.
    
    Args:
        name: Logger name (defaults to module name)
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    if name is None:
        name = __name__
        
    if log_dir is None:
        log_dir = os.environ.get('LOG_DIR', 'logs')
    
    return setup_logger(
        name=name,
        log_dir=log_dir,
        console_level=logging.INFO,
        file_level=logging.DEBUG,
        json_format=True
    )

# Global exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler to ensure all uncaught exceptions are logged."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Call the default handler for keyboard interrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
    logger = get_logger('error_logger')
    logger.error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )

# Set the global exception handler
sys.excepthook = handle_exception