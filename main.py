#!/usr/bin/env python3
"""
Main entry point for the NER Trip Extraction system.

This script demonstrates the usage of the trip parser to extract
departure and arrival cities from French sentences.
"""

import logging
from trip import TripParser
from trip.utils import setup_logging, format_trip_result


def main():
    """Run interactive trip extraction."""
    # Setup logging
    setup_logging(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info("Starting Trip Extraction Demo")

    # Initialize the parser (this will load the model)
    try:
        parser = TripParser()
    except Exception as e:
        logger.error(f"Failed to initialize parser: {e}")
        logger.error("Make sure you have installed the required dependencies:")
        logger.error("  pip install -e .")
        return 1

    print("\n" + "=" * 60)
    print("Trip Information Extraction")
    print("=" * 60)
    print("\nEntrez des phrases pour extraire les trajets.")
    print("Commandes: 'quit' ou 'exit' pour quitter\n")

    while True:
        try:
            # Prompt user for input
            sentence = input("✈️  Phrase > ").strip()

            if not sentence:
                continue

            # Check for exit commands
            if sentence.lower() in ["quit", "exit", "q"]:
                print("\n👋 Au revoir!")
                break

            # Parse the trip
            departure, arrival = parser.parse_trip(sentence)
            trip_str = format_trip_result(departure, arrival)

            print(f"➡️  Résultat: {trip_str}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except EOFError:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            logger.error(f"Error processing sentence: {e}")
            print(f"❌ Erreur: {e}\n")

    logger.info("Demo completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())
