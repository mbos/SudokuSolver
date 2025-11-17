# OCR Improvement Plan voor Sudoku Solver

**Document Datum**: 2025-01-17
**Huidige OCR Accuraatheid**: 84% (21/25 cijfers correct herkend)
**Doel**: >95% accuraatheid

## Executive Summary

Het Sudoku solver systeem heeft momenteel een accuraatheid van 84% in OCR digit recognition. De primaire bottleneck ligt bij het herkennen van 4 specifieke cijfers die wel visuele inhoud hebben maar niet door Tesseract worden herkend. Dit document presenteert een evidence-based verbeterplan gebaseerd op recente OCR research (2024-2025).

---

## Huidige Situatie Analyse

### OCR Failures op testplaatje.png

| Cell | Position | Werkelijk Cijfer | OCR Output | Fill Ratio | Status |
|------|----------|------------------|------------|------------|--------|
| (0,7) | Rij 1, Kolom 8 | 6 | 0 | 12.75% | Gemist |
| (1,4) | Rij 2, Kolom 5 | 9 | 0 | 12.38% | Gemist |
| (8,3) | Rij 9, Kolom 4 | 9 | 0 | 12.44% | Gemist |
| (8,6) | Rij 9, Kolom 7 | 8 | 0 | 12.94% | Gemist |

**Observaties**:
- Alle gemiste cellen hebben >12% fill ratio (ver boven 3% drempel voor "leeg")
- `preprocess_cell()` detecteert correct dat er inhoud is
- Tesseract faalt bij recognition stap (lage confidence of geen output)
- Geen false positives (0% incorrecte detecties)

### Huidige Pipeline Limitaties

1. **Single OCR Engine**: Alleen Tesseract (met CNN fallback)
2. **Beperkte Preprocessing**: Basis thresholding en morphological operations
3. **Lage Upscaling**: Slechts 4x upscale voor Tesseract
4. **Geen Data Augmentation**: CNN getraind op originele MNIST zonder augmentatie
5. **Fixed Confidence Threshold**: 0.7 voor CNN zonder adaptieve logic

---

## Verbeteringsstrategieën

Gebaseerd op recent OCR research uit 2024-2025, presenteren we een multi-tier aanpak:

### TIER 1: Quick Wins (1-2 dagen implementatie)

#### 1.1 Enhanced Preprocessing Pipeline
**Evidence**: "Improving input image quality through preprocessing can increase OCR accuracy by 15–30%"

**Implementatie**:
```python
class EnhancedPreprocessor:
    def preprocess_cell(self, cell):
        # 1. Aggressive upscaling (8x-10x instead of 4x)
        upscaled = cv2.resize(cell, None, fx=10, fy=10,
                             interpolation=cv2.INTER_CUBIC)

        # 2. Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(upscaled)

        # 3. Bilateral filtering (preserve edges, reduce noise)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # 4. Multiple threshold methods
        _, otsu = cv2.threshold(denoised, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(denoised, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

        # 5. Combine thresholding methods
        combined = cv2.bitwise_or(otsu, adaptive)

        return combined
```

**Verwachte Impact**: +5-10% accuraatheid
**Prioriteit**: HOOG
**Risico**: Laag

#### 1.2 Tesseract Configuration Optimization
**Evidence**: "Adjusting PSM modes and confidence thresholds can dramatically improve digit recognition"

**Implementatie**:
```python
# Test multiple PSM modes for failed cells
psm_modes = [
    '--psm 10',  # Single character (current)
    '--psm 8',   # Single word
    '--psm 7',   # Single text line
    '--psm 13'   # Raw line (no OSD/layout)
]

# Adjust Tesseract parameters
tesseract_config = {
    'tessedit_char_whitelist': '123456789',
    'tessedit_char_blacklist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
    'load_system_dawg': 'F',
    'load_freq_dawg': 'F',
    'matcher_bad_match_pad': '0.15',  # Lower threshold
}
```

**Verwachte Impact**: +3-5% accuraatheid
**Prioriteit**: HOOG
**Risico**: Laag

#### 1.3 Lower CNN Confidence Threshold Strategically
**Implementatie**:
```python
def recognize_with_cnn_adaptive(self, cell_image):
    predictions = self.model.predict(input_data, verbose=0)
    digit = np.argmax(predictions[0])
    confidence = predictions[0][digit]

    # Adaptive thresholding based on digit
    # Some digits (like 6, 8, 9) are harder to distinguish
    thresholds = {
        6: 0.60,  # Lower for 6 (looks like 0)
        8: 0.60,  # Lower for 8 (looks like 3)
        9: 0.60,  # Lower for 9 (looks like 4)
        'default': 0.70
    }

    threshold = thresholds.get(digit, thresholds['default'])

    if confidence > threshold:
        return int(digit)
    return 0
```

**Verwachte Impact**: +2-4% accuraatheid
**Prioriteit**: MEDIUM
**Risico**: MEDIUM (mogelijk meer false positives)

---

### TIER 2: Medium-term Improvements (1 week implementatie)

#### 2.1 Ensemble OCR System
**Evidence**: "Ensemble methods achieved 75.77% accuracy where single engines got 48.66% and 35.71%"
**Evidence**: "33% error reduction compared to best single result"

**Implementatie**:
```python
class EnsembleOCR:
    def __init__(self):
        self.engines = [
            TesseractEngine(),
            CNNEngine(),
            EasyOCREngine(),  # Alternative OCR
        ]

    def recognize_with_voting(self, cell):
        results = []

        for engine in self.engines:
            digit, confidence = engine.recognize(cell)
            results.append((digit, confidence))

        # Confidence-weighted voting
        votes = {}
        for digit, conf in results:
            if digit != 0:
                votes[digit] = votes.get(digit, 0) + conf

        if not votes:
            return 0

        # Return digit with highest confidence sum
        return max(votes.items(), key=lambda x: x[1])[0]
```

**Components**:
1. **Primary**: Current CNN + Tesseract
2. **Secondary**: EasyOCR (modern deep learning OCR)
3. **Tertiary**: PaddleOCR (alternative)

**Voting Strategy**:
- Confidence-weighted voting
- Require 2/3 agreement for high confidence
- Fall back to highest confidence if disagreement

**Verwachte Impact**: +8-15% accuraatheid
**Prioriteit**: HOOG
**Risico**: MEDIUM (dependencies, performance)

#### 2.2 Improved CNN Training with Data Augmentation
**Evidence**: "CNN accuracy improved from 96% to 99.67% after applying augmentation"

**Implementatie**:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create augmentation pipeline
datagen = ImageDataGenerator(
    rotation_range=10,           # ±10 degrees
    width_shift_range=0.1,       # 10% shift
    height_shift_range=0.1,      # 10% shift
    shear_range=0.1,            # Shear transformation
    zoom_range=0.1,             # Zoom in/out
    fill_mode='constant',        # Fill with black
    cval=0
)

# Add elastic deformation (shown to work well for digits)
def elastic_transform(image, alpha=36, sigma=4):
    """Apply elastic deformation"""
    # Implementation of elastic transformation
    pass

# Train with augmented data
model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=10,  # More epochs with augmentation
    validation_data=(x_test, y_test)
)
```

**Augmentation Techniques**:
1. Rotation (±10 degrees)
2. Translation (±10%)
3. Shear transformation
4. Elastic deformation (proven effective for MNIST)
5. Zoom (90%-110%)

**Verwachte Impact**: +5-8% accuraatheid
**Prioriteit**: MEDIUM
**Risico**: Laag

#### 2.3 Template Matching Fallback
**Evidence**: "Template matching reported as effective alternative for failed digit recognition"

**Implementatie**:
```python
def match_template(self, cell):
    """
    Use template matching as last resort for failed OCR.
    Pre-create templates from correctly recognized digits.
    """
    best_match = None
    best_score = 0

    for digit in range(1, 10):
        templates = self.digit_templates[digit]

        for template in templates:
            # Try multiple scales
            for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
                resized = cv2.resize(template, None, fx=scale, fy=scale)
                result = cv2.matchTemplate(cell, resized, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)

                if score > best_score:
                    best_score = score
                    best_match = digit

    if best_score > 0.7:  # Confidence threshold
        return best_match
    return 0
```

**Verwachte Impact**: +3-5% accuraatheid (voor edge cases)
**Prioriteit**: LOW
**Risico**: Laag

---

### TIER 3: Advanced Improvements (2-3 weken implementatie)

#### 3.1 Integration van Multimodal LLM
**Evidence**: "GPT-4o and Claude 3.7 Sonnet achieve 82–90% accuracy vs traditional 50–70%"

**Implementatie**:
```python
import anthropic

def recognize_with_llm(self, cell_image):
    """Use Claude Sonnet for difficult cases"""
    client = anthropic.Anthropic()

    # Convert cell to base64
    _, buffer = cv2.imencode('.png', cell_image)
    image_data = base64.b64encode(buffer).decode('utf-8')

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "What single digit (1-9) is shown in this image? Reply with only the digit, or 0 if empty."
                }
            ],
        }],
    )

    try:
        digit = int(message.content[0].text.strip())
        if 1 <= digit <= 9:
            return digit
    except:
        pass

    return 0
```

**Strategie**:
- Gebruik alleen voor cellen waar alle andere methodes falen
- Cost-effective door selectieve toepassing
- Hoge accuraatheid voor edge cases

**Verwachte Impact**: +2-5% accuraatheid (voor moeilijke gevallen)
**Prioriteit**: LOW
**Risico**: MEDIUM (cost, API dependency)
**Opmerking**: Alleen voor productie, niet voor development/testing

#### 3.2 Self-Supervised Fine-tuning
**Evidence**: "Self-supervised pretraining yields significant boosts in recognition accuracy"

**Implementatie**:
```python
# Fine-tune CNN on Sudoku-specific images
def create_sudoku_dataset():
    """
    Generate training data from successfully recognized cells
    and manual corrections of failed cases.
    """
    dataset = []

    # Use cells from testplaatje.png and other Sudoku images
    # Label them correctly (including the 4 failed ones)

    return dataset

# Fine-tune existing MNIST model
def fine_tune_on_sudoku_data():
    sudoku_data = create_sudoku_dataset()

    # Unfreeze last layers
    for layer in model.layers[-4:]:
        layer.trainable = True

    # Train with low learning rate
    model.compile(optimizer=Adam(lr=0.0001), ...)
    model.fit(sudoku_data, epochs=20)
```

**Verwachte Impact**: +5-10% accuraatheid (domain-specific)
**Prioriteit**: MEDIUM
**Risico**: MEDIUM (requires labeled Sudoku data)

---

## Testing Framework

### Test Dataset Creatie

#### 1. Ground Truth Dataset
```python
# Create test_dataset.json
{
    "images": [
        {
            "filename": "testplaatje.png",
            "ground_truth": [
                [0,0,0,0,0,0,9,6,5],
                [0,0,0,1,9,0,0,0,0],
                [0,0,0,2,0,0,0,0,8],
                [1,0,0,7,6,0,0,0,0],
                [0,9,5,0,0,0,0,0,0],
                [0,0,7,0,1,0,5,3,0],
                [0,0,3,0,2,1,0,0,0],
                [7,0,0,0,0,0,1,5,0],
                [6,0,0,9,0,0,8,0,0]
            ]
        }
        // ... more test images
    ]
}
```

**Dataset Samenstelling**:
- **Minimaal**: 10 verschillende Sudoku images
- **Ideaal**: 50-100 images met variërende:
  - Font types
  - Image quality (scans, photos, screenshots)
  - Lighting conditions
  - Grid line thickness
  - Digit sizes

#### 2. Synthetic Test Data Generatie
```python
def generate_synthetic_sudoku_images():
    """
    Generate test images with known ground truth
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create Sudoku images with various fonts
    fonts = ['Arial', 'Times New Roman', 'Courier', 'Helvetica']

    for font_name in fonts:
        for quality in ['high', 'medium', 'low']:
            # Generate image
            # Add noise, blur, rotation based on quality
            # Save with ground truth
            pass
```

### Automated Testing Suite

```python
# tests/test_ocr_accuracy.py

import pytest
import json
import numpy as np
from src.ocr import DigitRecognizer

class OCRTestSuite:
    """Comprehensive OCR testing framework"""

    def __init__(self, test_dataset_path='test_dataset.json'):
        with open(test_dataset_path) as f:
            self.test_data = json.load(f)

    def test_overall_accuracy(self, recognizer):
        """Test overall digit recognition accuracy"""
        total_digits = 0
        correct_digits = 0

        for test_case in self.test_data['images']:
            # Load image and extract cells
            cells = self.extract_cells(test_case['filename'])

            # Run OCR
            detected_grid, _ = recognizer.recognize_grid(cells)
            ground_truth = np.array(test_case['ground_truth'])

            # Compare only filled cells
            mask = ground_truth != 0
            total_digits += np.sum(mask)
            correct_digits += np.sum(detected_grid[mask] == ground_truth[mask])

        accuracy = correct_digits / total_digits
        return accuracy

    def test_per_digit_accuracy(self, recognizer):
        """Test accuracy per digit (1-9)"""
        digit_stats = {i: {'total': 0, 'correct': 0} for i in range(1, 10)}

        for test_case in self.test_data['images']:
            cells = self.extract_cells(test_case['filename'])
            detected_grid, _ = recognizer.recognize_grid(cells)
            ground_truth = np.array(test_case['ground_truth'])

            for digit in range(1, 10):
                mask = ground_truth == digit
                digit_stats[digit]['total'] += np.sum(mask)
                digit_stats[digit]['correct'] += np.sum(
                    detected_grid[mask] == ground_truth[mask]
                )

        # Calculate accuracy per digit
        results = {}
        for digit, stats in digit_stats.items():
            if stats['total'] > 0:
                results[digit] = stats['correct'] / stats['total']

        return results

    def test_false_positive_rate(self, recognizer):
        """Test false positive rate (detecting digit in empty cell)"""
        total_empty = 0
        false_positives = 0

        for test_case in self.test_data['images']:
            cells = self.extract_cells(test_case['filename'])
            detected_grid, _ = recognizer.recognize_grid(cells)
            ground_truth = np.array(test_case['ground_truth'])

            empty_mask = ground_truth == 0
            total_empty += np.sum(empty_mask)
            false_positives += np.sum(detected_grid[empty_mask] != 0)

        fp_rate = false_positives / total_empty
        return fp_rate

    def test_false_negative_rate(self, recognizer):
        """Test false negative rate (missing actual digits)"""
        total_filled = 0
        false_negatives = 0

        for test_case in self.test_data['images']:
            cells = self.extract_cells(test_case['filename'])
            detected_grid, has_content = recognizer.recognize_grid(cells)
            ground_truth = np.array(test_case['ground_truth'])

            filled_mask = ground_truth != 0
            total_filled += np.sum(filled_mask)
            # False negative: has content but OCR returned 0
            false_negatives += np.sum(
                (has_content[filled_mask]) & (detected_grid[filled_mask] == 0)
            )

        fn_rate = false_negatives / total_filled
        return fn_rate

    def benchmark_performance(self, recognizer):
        """Benchmark speed and accuracy"""
        import time

        times = []
        accuracies = []

        for test_case in self.test_data['images']:
            cells = self.extract_cells(test_case['filename'])

            start = time.time()
            detected_grid, _ = recognizer.recognize_grid(cells)
            elapsed = time.time() - start
            times.append(elapsed)

            ground_truth = np.array(test_case['ground_truth'])
            mask = ground_truth != 0
            accuracy = np.sum(
                detected_grid[mask] == ground_truth[mask]
            ) / np.sum(mask)
            accuracies.append(accuracy)

        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        }

    def generate_confusion_matrix(self, recognizer):
        """Generate confusion matrix for digit recognition"""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        y_true = []
        y_pred = []

        for test_case in self.test_data['images']:
            cells = self.extract_cells(test_case['filename'])
            detected_grid, _ = recognizer.recognize_grid(cells)
            ground_truth = np.array(test_case['ground_truth'])

            # Only compare filled cells
            mask = ground_truth != 0
            y_true.extend(ground_truth[mask].flatten())
            y_pred.extend(detected_grid[mask].flatten())

        cm = confusion_matrix(y_true, y_pred, labels=range(1, 10))

        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('OCR Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(1, 10)
        plt.xticks(tick_marks-1, tick_marks)
        plt.yticks(tick_marks-1, tick_marks)
        plt.ylabel('True Digit')
        plt.xlabel('Predicted Digit')
        plt.savefig('confusion_matrix.png')

        return cm
```

### Comparison Testing

```python
def compare_ocr_methods():
    """Compare different OCR implementations"""

    methods = {
        'baseline': DigitRecognizer(use_tesseract=True),
        'enhanced_preprocessing': EnhancedDigitRecognizer(),
        'ensemble': EnsembleOCR(),
        'cnn_augmented': AugmentedCNNRecognizer(),
    }

    suite = OCRTestSuite()
    results = {}

    for name, recognizer in methods.items():
        print(f"\nTesting {name}...")
        results[name] = {
            'overall_accuracy': suite.test_overall_accuracy(recognizer),
            'per_digit': suite.test_per_digit_accuracy(recognizer),
            'false_positive_rate': suite.test_false_positive_rate(recognizer),
            'false_negative_rate': suite.test_false_negative_rate(recognizer),
            'performance': suite.benchmark_performance(recognizer),
        }

    # Generate comparison report
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    for name, metrics in results.items():
        print(f"\n{name.upper()}")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.2%}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.2%}")
        print(f"  Avg Time: {metrics['performance']['avg_time']:.3f}s")

    return results
```

### Continuous Testing

```python
# pytest integration
def test_ocr_baseline_accuracy():
    """Ensure OCR accuracy doesn't regress below baseline"""
    recognizer = DigitRecognizer(use_tesseract=True)
    suite = OCRTestSuite()

    accuracy = suite.test_overall_accuracy(recognizer)

    # Baseline should be at least 84%
    assert accuracy >= 0.84, f"OCR accuracy {accuracy:.2%} below baseline 84%"

def test_ocr_target_accuracy():
    """Test if target accuracy is reached"""
    recognizer = ImprovedDigitRecognizer()
    suite = OCRTestSuite()

    accuracy = suite.test_overall_accuracy(recognizer)

    # Target is 95%+
    assert accuracy >= 0.95, f"OCR accuracy {accuracy:.2%} below target 95%"

# Run with: pytest tests/test_ocr_accuracy.py -v
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Create test dataset met ground truth (10+ images)
- [ ] Implement testing framework
- [ ] Establish baseline metrics
- [ ] Implement TIER 1.1: Enhanced Preprocessing
- [ ] Implement TIER 1.2: Tesseract Optimization
- [ ] Run comparison tests

**Exit Criteria**: >88% accuracy, test framework operational

### Phase 2: Ensemble & Training (Week 2)
- [ ] Implement TIER 2.1: Ensemble OCR System
- [ ] Implement TIER 2.2: CNN Data Augmentation
- [ ] Expand test dataset to 25+ images
- [ ] Run comprehensive benchmarks
- [ ] Tune confidence thresholds

**Exit Criteria**: >93% accuracy, <2% false positive rate

### Phase 3: Advanced & Fine-tuning (Week 3)
- [ ] Implement TIER 2.3: Template Matching
- [ ] Collect Sudoku-specific training data
- [ ] Implement TIER 3.2: Fine-tuning on Sudoku data
- [ ] Optional: Implement TIER 3.1 (LLM integration)
- [ ] Final testing and optimization
- [ ] Documentation update

**Exit Criteria**: >95% accuracy, production ready

---

## Success Metrics

### Primary Metrics
- **Overall Accuracy**: 84% → 95%+ (target)
- **False Negative Rate**: 16% → <5%
- **False Positive Rate**: 0% → <2% (acceptable trade-off)

### Secondary Metrics
- **Per-digit Accuracy**: All digits >90%
- **Processing Time**: <2s per puzzle (real-time capable)
- **Robustness**: Works on varied image qualities

### Test Coverage
- ✅ Multiple image sources (scan, photo, screenshot)
- ✅ Various font types and sizes
- ✅ Different lighting conditions
- ✅ Edge cases (faded, rotated, noisy images)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Ensemble increases latency | Medium | Medium | Parallel processing, selective use |
| False positives increase | Medium | High | Careful threshold tuning, validation |
| Overfitting on test data | Low | Medium | Separate validation set, cross-validation |
| External API costs (LLM) | Low | Low | Use only for failed cases, budget limits |
| Model size increases | Low | Low | Model compression, quantization |

---

## Cost-Benefit Analysis

### Quick Wins (TIER 1)
- **Effort**: 1-2 days
- **Expected Gain**: +10-15% accuracy
- **ROI**: Excellent
- **Recommendation**: Implement immediately

### Medium-term (TIER 2)
- **Effort**: 1 week
- **Expected Gain**: +15-25% total accuracy
- **ROI**: Very Good
- **Recommendation**: High priority

### Advanced (TIER 3)
- **Effort**: 2-3 weeks
- **Expected Gain**: +5-10% additional accuracy
- **ROI**: Good for edge cases
- **Recommendation**: Optional, based on requirements

---

## Appendix: Tools and Libraries

### Recommended OCR Libraries
```bash
# Current
pip install pytesseract opencv-python tensorflow

# Additional for ensemble
pip install easyocr
pip install paddlepaddle paddleocr

# For LLM integration (optional)
pip install anthropic
```

### Testing Tools
```bash
pip install pytest pytest-benchmark
pip install scikit-learn matplotlib seaborn
```

### Performance Profiling
```bash
pip install py-spy memory_profiler
```

---

## References

1. "Improving OCR Quality with Advanced Techniques" - SparkCo.ai (2024)
2. "The Definitive Guide to OCR Accuracy" - Medium (2025)
3. "OpenCV Sudoku Solver and OCR" - PyImageSearch (2020)
4. "Ensemble OCR Methods" - Ocromore Research (2024)
5. "Data Augmentation for MNIST" - Journal of Big Data (2024)
6. "Improving Tesseract OCR with Convolution Preprocessing" - MDPI (2024)

---

**Document Owner**: Sudoku Solver Development Team
**Last Updated**: 2025-01-17
**Version**: 1.0
