# Trip Information Extraction

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Natural Language Processing system that extracts travel information (departure and arrival cities) from French text using CamemBERT Named Entity Recognition (NER).

## 🎯 Features

- **French NER**: Uses CamemBERT for accurate French language entity recognition
- **Trip Parsing**: Intelligent extraction of departure and arrival cities
- **Keyword Analysis**: Context-aware parsing using linguistic patterns
- **Robust Error Handling**: Graceful handling of edge cases and errors
- **Well-Tested**: Comprehensive test suite with pytest
- **Type Hints**: Full type annotations for better IDE support

## 📋 Requirements

- Python 3.10 or higher
- ~4GB disk space (for model weights)
- Internet connection (first run only, to download the model)

## 🚀 Installation

### Using pip (recommended)

```bash
# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Using a virtual environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e ".[dev]"
```

## 💻 Usage

### Command Line

Run the demo script:

```bash
python main.py
```

Or use the installed command:

```bash
trip
```

### As a Library

```python
from trip import TripParser
from trip.utils import setup_logging, format_trip_result

# Setup logging (optional)
setup_logging()

# Initialize the parser
parser = TripParser()

# Extract trip information
text = "Je veux aller à Lille depuis Paris"
departure, arrival = parser.parse_trip(text)

print(f"Départ: {departure}")    # Paris
print(f"Arrivée: {arrival}")      # Lille

# Or use the formatter
result = format_trip_result(departure, arrival)
print(result)  # Paris → Lille
```

### Advanced Usage

```python
from trip import NERExtractor, TripParser

# Use a custom NER model
ner = NERExtractor(model_name="your-custom-model")

# Create parser with custom NER
parser = TripParser(ner_extractor=ner)

# Extract all entities
entities = ner.extract_entities("Jean va de Paris à Lyon")
for entity in entities:
    print(f"{entity['word']} ({entity['entity_group']}): {entity['score']:.2f}")

# Extract only locations
locations = ner.extract_locations("Train de Marseille à Bordeaux")
print(locations)  # ['Marseille', 'Bordeaux']
```

## 📁 Project Structure

```
trip/
├── src/
│   └── trip/
│       ├── __init__.py          # Package initialization
│       ├── __main__.py          # CLI entry point
│       ├── ner_extractor.py     # NER extraction logic
│       ├── trip_parser.py       # Trip parsing logic
│       └── utils.py             # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_ner_extractor.py   # NER tests
│   ├── test_trip_parser.py     # Parser tests
│   └── test_utils.py            # Utility tests
├── main.py                      # Demo script
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## 🛠️ Development

### Setting up for development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/
ruff check src/ tests/ --fix

# Run type checking
mypy src/
```

### Code Quality Tools

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Pytest**: Testing framework

## 📝 Examples

```python
# Example 1: Basic usage
parser = TripParser()
result = parser.parse_trip("Je veux aller à Lille depuis Paris")
# => ('Paris', 'Lille')

# Example 2: Different phrasings
examples = [
    "Je veux prendre le train de Montpellier à Paris",
    "Train Paris → Strasbourg",
    "Je pars demain de Lyon pour Marseille",
    "Vol Paris Marseille demain",
]

for text in examples:
    departure, arrival = parser.parse_trip(text)
    print(f"{text} => {departure} → {arrival}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [CamemBERT](https://camembert-model.fr/) for the French NER model
- [Hugging Face Transformers](https://huggingface.co/transformers/) for the NLP toolkit
- [Jean-Baptiste/camembert-ner](https://huggingface.co/Jean-Baptiste/camembert-ner) for the pre-trained model

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🐛 Known Issues

- First run requires internet connection to download model (~250MB)
- Model loading can take 10-30 seconds depending on hardware
- Best results with clear, well-structured French sentences

## 🗺️ Roadmap

- [ ] Add support for more complex trip patterns
- [ ] Implement caching for faster model loading
- [ ] Add REST API interface
- [ ] Support for additional languages
- [ ] Fine-tune model on trip-specific data
- [ ] Add confidence scores to results
