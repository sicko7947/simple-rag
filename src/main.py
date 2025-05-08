"""
Main entry point for the RAG application
"""
import os
import logging
from dotenv import load_dotenv
from src.user_interface.cli import run_cli
from src.user_interface.web import run_web_app

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    logger.info("Starting RAG application")
    
    # Determine which interface to use (CLI or web)
    interface = os.getenv("INTERFACE", "cli").lower()
    
    if interface == "cli":
        run_cli()
    elif interface == "web":
        run_web_app()
    else:
        logger.error("Invalid interface specified: %s", interface)
        print("Error: Invalid interface specified. Please set INTERFACE to 'cli' or 'web'")

if __name__ == "__main__":
    main()