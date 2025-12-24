#!/usr/bin/env python3
"""
Main entry point for the NER Trip Extraction system.

This script demonstrates the usage of the trip parser to extract
departure and arrival cities from French sentences.
"""

import logging
import sys

from trip_parser import TripParser
from trip_parser.exceptions import ModelNotFoundError, TripExtractionError
from trip_parser.utils import format_trip_result, setup_logging

# Setup module logger
logger = logging.getLogger(__name__)


def print_banner():
    """Print the application banner."""
    banner = "\n" + "=" * 60 + "\n"
    banner += "Trip Information Extraction v0.1.0\n"
    banner += "=" * 60 + "\n"
    banner += "\nEntrez des phrases pour extraire les trajets.\n"
    banner += "Commandes: 'quit' ou 'exit' pour quitter\n"
    print(banner)


def main():
    """
    Run interactive trip extraction.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Setup logging
    try:
        setup_logging(level=logging.INFO)
    except Exception as e:
        print(f"Failed to setup logging: {e}", file=sys.stderr)
        return 1

    logger.info("Starting Trip Extraction Demo v0.1.0")

    # Initialize the parser (this will load the models)
    try:
        logger.info("Loading models...")
        parser = TripParser()
        logger.info("Models loaded successfully")

    except ModelNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.error("Please train the model first:")
        logger.error("  python scripts/train.py")
        return 1

    except TripExtractionError as e:
        logger.error(f"Failed to initialize parser: {e}")
        logger.error("Make sure you have installed the required dependencies:")
        logger.error("  pip install -e .")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}", exc_info=True)
        return 1

    # Print banner
    print_banner()

    # Main interaction loop
    while True:
        try:
            # Prompt user for input
            sentence = input("✈️  Phrase > ").strip()

            if not sentence:
                continue

            # Check for exit commands
            if sentence.lower() in ["quit", "exit", "q"]:
                logger.info("User requested exit")
                print("\n👋 Au revoir!")
                break

            # Parse the trip
            logger.debug(f"Processing input: {sentence}")
            departure, arrival = parser.parse_trip(sentence)
            trip_str = format_trip_result(departure, arrival)

            print(f"➡️  Résultat: {trip_str}\n")

        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
            print("\n\n👋 Au revoir!")
            break

        except EOFError:
            logger.info("EOF received")
            print("\n\n👋 Au revoir!")
            break

        except TripExtractionError as e:
            logger.warning(f"Trip extraction error: {e}")
            print(f"⚠️  Avertissement: {e}\n")

        except Exception as e:
            logger.error(f"Error processing sentence: {e}", exc_info=True)
            print(f"❌ Erreur: {e}\n")

    logger.info("Demo completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
