# Test Images Archive

Deze directory bevat test en debug images die gegenereerd zijn tijdens development.

## Inhoud

### `synthetic_tests/`
15 synthetische Sudoku test images gegenereerd met `generate_test_dataset.py`:
- Verschillende quality levels (high, medium, low)
- Verschillende difficulty levels (easy, medium, hard)
- Verschillende fonts en variaties (rotation, perspective, noise)

### Debug Images (Root)
- `failed_*.png` - Preprocessed images van digits die fout herkend werden
- `preprocessed_*.png` - Preprocessing visualisaties voor analyse

## Gebruik

Deze images werden gebruikt voor:
- OCR accuracy testing
- Preprocessing pipeline development
- Error analysis
- CNN training validation

Ze zijn gearchiveerd voor referentie maar niet nodig voor normale productie gebruik.

## Verwijderen

Deze directory kan veilig verwijderd worden zonder impact op het systeem:
```bash
rm -rf test_images_archive/
```
