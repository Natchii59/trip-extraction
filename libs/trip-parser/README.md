# Trip Parser Library

A NER-based trip extraction library using CamemBERT for French text.

## Description

This library provides functionality to extract departure and arrival cities from French text using Named Entity Recognition (NER) with CamemBERT models.

## Installation

```bash
cd libs/trip-parser
pip install -e .
```

## Usage

### As a Library

```python
from trip_parser import TripParser

parser = TripParser()
result = parser.extract_trip("Je vais de Paris à Lyon")
print(result.departure)  # Paris
print(result.arrival)    # Lyon
```

### Training

```bash
nx run trip-parser:train
```

### Demo

```bash
nx run trip-parser:demo
```

## Project Structure

- `src/trip_parser/` - Main library code
- `scripts/` - Training and demo scripts
- `models/` - Trained models (not in git)
- `datasets/` - Training datasets
