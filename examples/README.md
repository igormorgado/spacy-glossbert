# GlossBERT WSD Examples

This directory contains examples demonstrating how to use the GlossBERT WSD spaCy component.

## Examples

1. **basic_usage.py**: Simple demonstration of the component in a pipeline
2. **advanced_usage.py**: Advanced usage with custom configuration, visualization, and detailed analysis

## Running the Examples

Make sure you have installed the package and its dependencies first:

```bash
# Install from PyPI
pip install glossbert-wsd

# Or install from the current directory in development mode
pip install -e ..
```

You'll also need to download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

Then run the examples:

```bash
python basic_usage.py
python advanced_usage.py
```

## Visualization

The example with visualization will work best in a Jupyter notebook environment or when served via a web server. 
For command line usage, the examples will print information about the disambiguated word senses without actual visualization.

To use in a Jupyter notebook:

```python
from glossbert_wsd import visualize_wsd
from advanced_usage import custom_visualization

# Use the built-in visualization
visualize_wsd(doc)

# Or use the custom visualization with colored POS tags
custom_visualization(doc)
``` 