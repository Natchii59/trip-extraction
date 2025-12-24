#!/usr/bin/env python3
"""
Entry point for running the Trip Parser API server.

This script starts the FastAPI application using uvicorn.
"""

import argparse
import logging
import sys

import uvicorn

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the Trip Parser API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (localhost:8000)
  python scripts/run_api.py

  # Run on a specific host and port
  python scripts/run_api.py --host 0.0.0.0 --port 8080

  # Run with auto-reload for development
  python scripts/run_api.py --reload

  # Or use the installed command
  trip-api --port 8080
        """,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: False)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, only for production)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level (default: info)",
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the API server.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Trip Parser API Server")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info("=" * 60)

    try:
        # Start the server
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,  # Workers incompatible with reload
            log_level=args.log_level,
            access_log=True,
        )
        return 0

    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        return 0

    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
