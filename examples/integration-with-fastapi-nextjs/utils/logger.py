from loguru import logger
import json
import os
import sys
from uuid import UUID

class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

class LogConfig:
    """Centralized logging configuration for the application"""
    
    # Log file paths from environment variables
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "api_logs.json")
    ERROR_LOG_PATH = os.getenv("ERROR_LOG_PATH", "api_error_logs.json")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @staticmethod
    def json_sink(message):
        """Formats and writes log messages in the required JSON format"""
        record = message.record
        
        # Create the exact JSON structure wanted
        log_entry = {
            "timestamp": record["time"].timestamp(),
            "file_name": record["file"].name,
            "line": record["line"],
            "function": record["function"],
            "message": record["message"],
            "module": record["module"],
            "name": record["name"],
            "level": record["level"].name
        }
        
        # Add all extra fields
        log_entry.update(record["extra"])
        
        # Write to main log file
        with open(LogConfig.LOG_FILE_PATH, "a") as f:
            f.write(json.dumps(log_entry, cls=UUIDEncoder) + "\n")
            
        # Also write errors to separate file
        if record["level"].name in ["ERROR", "CRITICAL"]:
            with open(LogConfig.ERROR_LOG_PATH, "a") as f:
                f.write(json.dumps(log_entry, cls=UUIDEncoder) + "\n")

    @classmethod
    def setup_logging(cls):
        """Initialize and configure the logger"""
        # Remove default handler
        logger.remove()
        
        # Create a sink function that formats extra fields
        def console_sink(message):
            record = message.record
            # Format the base message
            formatted = f"{record['time'].strftime('%B %d, %Y - %H:%M:%S!UTC')} | {record['level'].name} | {record['message']}"
            
            # Add extra fields if they exist
            if record["extra"]:
                extras = " | ".join([f"{k}={v}" for k, v in record["extra"].items() if k])
                if extras:
                    formatted += f" | {extras}"
                    
            # Print to stderr
            print(formatted, file=sys.stderr)
            
        # Add custom sink for console output
        logger.add(console_sink, level=LogConfig.LOG_LEVEL)
        
        # Add our JSON sink
        logger.add(cls.json_sink, level=LogConfig.LOG_LEVEL)
        
        return logger

# Create configured logger instance
def get_logger():
    """Get the configured logger instance"""
    LogConfig.setup_logging()
    return logger

# Export the configured logger
app_logger = get_logger()
