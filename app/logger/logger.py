import logging
from pathlib import Path
from app.config.config import config

def get_logger(
    name: str,
    level: str = 'INFO',
    log_file: str | Path | None = None
) -> logging.Logger:
    '''
    Return a configured logger instance.
    '''
    
    # Use provided level or fallback to config
    level = level or config.log_level
    
    # Adjust level based on environment
    if config.environment == 'development':
        # show more detail (DEBUG)
        if level.upper() == 'INFO':
            level = 'DEBUG'
        
        elif config.environment == 'production':
            # only log warnings and above
            if level.upper() in ('INFO', 'DEBUG'):
                level = 'WARNING'
    
    # Get or create a logger
    logger = logging.getLogger(name)
    # Set the logging level, convert to uppercase 
    logger.setLevel(level.upper())
    
    # Configure handlers only once to prevent log duplication
    if not logger.handlers:
        
        # Create a formatter that defines the log message format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        
        # Create a handler that writes logs to the console (terminal)
        stream_handler = logging.StreamHandler()
        
        # Apply the formatter to the console handler
        stream_handler.setFormatter(formatter)
        
        # Add the console handler to the logger
        logger.addHandler(stream_handler)        

        # If a log_file was provided, also write logs to that file
        if log_file:
            
            # Convert the file path to a Path object
            log_path = Path(log_file)
            
            # Create parent directories if they do not exist
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a handler that writes logs to the file
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False
        
    return logger