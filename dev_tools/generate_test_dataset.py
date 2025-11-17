#!/usr/bin/env python3
"""
Generate synthetic Sudoku images for testing OCR accuracy.
Creates images with various fonts, qualities, and transformations.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
import random
from src.solver import SudokuSolver


class SudokuImageGenerator:
    """Generate synthetic Sudoku images with ground truth."""

    def __init__(self, output_dir="test_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Available fonts (using system fonts)
        self.fonts = self._get_available_fonts()

    def _get_available_fonts(self):
        """Get list of available system fonts."""
        font_paths = [
            # macOS fonts
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Times.ttc",
            "/System/Library/Fonts/Courier.dfont",
            "/Library/Fonts/Arial.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            # Common Linux fonts
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

        available = []
        for path in font_paths:
            if os.path.exists(path):
                available.append(path)

        # Fallback to default if no fonts found
        if not available:
            available.append(None)  # PIL will use default

        return available

    def generate_puzzle(self, difficulty="medium"):
        """
        Generate a valid Sudoku puzzle.

        Args:
            difficulty: easy (35-45 clues), medium (27-35), hard (22-27)

        Returns:
            9x9 numpy array with puzzle (0 = empty)
        """
        # Start with a solved Sudoku
        solver = SudokuSolver()

        # Create a filled grid by solving an empty grid with random order
        base_grid = np.zeros((9, 9), dtype=int)
        solver.load_puzzle(base_grid)

        # Fill first row randomly
        first_row = list(range(1, 10))
        random.shuffle(first_row)
        for col, digit in enumerate(first_row):
            solver.grid[0, col] = digit

        solver.solve()
        full_grid = solver.get_solution()

        # Remove cells based on difficulty
        clue_ranges = {
            'easy': (35, 45),
            'medium': (27, 35),
            'hard': (22, 27)
        }

        min_clues, max_clues = clue_ranges.get(difficulty, (27, 35))
        num_clues = random.randint(min_clues, max_clues)

        # Create puzzle by removing cells
        puzzle = full_grid.copy()
        cells = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(cells)

        for i in range(81 - num_clues):
            row, col = cells[i]
            puzzle[row, col] = 0

        return puzzle

    def render_sudoku_image(
        self,
        puzzle: np.ndarray,
        font_path: str = None,
        image_size: int = 540,
        quality: str = "high"
    ) -> Image.Image:
        """
        Render Sudoku puzzle as PIL Image.

        Args:
            puzzle: 9x9 numpy array
            font_path: Path to TTF font file
            image_size: Size of output image in pixels
            quality: 'high', 'medium', or 'low'

        Returns:
            PIL Image
        """
        # Create white background
        img = Image.new('RGB', (image_size, image_size), 'white')
        draw = ImageDraw.Draw(img)

        cell_size = image_size // 9

        # Draw grid lines
        line_width = 2 if quality != "low" else 1
        thick_line_width = 4 if quality == "high" else 3

        for i in range(10):
            thickness = thick_line_width if i % 3 == 0 else line_width
            # Horizontal lines
            draw.line(
                [(0, i * cell_size), (image_size, i * cell_size)],
                fill='black',
                width=thickness
            )
            # Vertical lines
            draw.line(
                [(i * cell_size, 0), (i * cell_size, image_size)],
                fill='black',
                width=thickness
            )

        # Load font
        font_size = int(cell_size * 0.6)
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Draw digits
        for row in range(9):
            for col in range(9):
                digit = puzzle[row, col]
                if digit != 0:
                    text = str(digit)

                    # Get text bounding box
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Center text in cell
                    x = col * cell_size + (cell_size - text_width) // 2
                    y = row * cell_size + (cell_size - text_height) // 2

                    draw.text((x, y), text, fill='black', font=font)

        return img

    def apply_degradation(
        self,
        img: Image.Image,
        quality: str = "high"
    ) -> Image.Image:
        """
        Apply image degradation effects.

        Args:
            img: PIL Image
            quality: 'high', 'medium', or 'low'

        Returns:
            Degraded PIL Image
        """
        # Convert to numpy for OpenCV processing
        img_np = np.array(img)

        if quality == "medium":
            # Add slight blur
            img_np = cv2.GaussianBlur(img_np, (3, 3), 0)

            # Add noise
            noise = np.random.normal(0, 5, img_np.shape).astype(np.uint8)
            img_np = cv2.add(img_np, noise)

            # Slightly reduce contrast
            img_np = cv2.convertScaleAbs(img_np, alpha=0.9, beta=5)

        elif quality == "low":
            # More aggressive blur
            img_np = cv2.GaussianBlur(img_np, (5, 5), 1)

            # More noise
            noise = np.random.normal(0, 15, img_np.shape).astype(np.uint8)
            img_np = cv2.add(img_np, noise)

            # Reduce contrast more
            img_np = cv2.convertScaleAbs(img_np, alpha=0.8, beta=10)

            # Add JPEG compression artifacts
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            _, encimg = cv2.imencode('.jpg', img_np, encode_param)
            img_np = cv2.imdecode(encimg, 1)

        return Image.fromarray(img_np)

    def apply_transformations(
        self,
        img: Image.Image,
        rotation: float = 0,
        perspective: bool = False
    ) -> Image.Image:
        """
        Apply geometric transformations.

        Args:
            img: PIL Image
            rotation: Rotation angle in degrees
            perspective: Apply perspective transformation

        Returns:
            Transformed PIL Image
        """
        if rotation != 0:
            img = img.rotate(rotation, fillcolor='white', expand=False)

        if perspective:
            # Convert to numpy for perspective transform
            img_np = np.array(img)
            h, w = img_np.shape[:2]

            # Create slight perspective effect
            offset = int(h * 0.05)
            src_points = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ])
            dst_points = np.float32([
                [offset, offset],
                [w - offset, 0],
                [w, h - offset],
                [0, h]
            ])

            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            img_np = cv2.warpPerspective(img_np, matrix, (w, h),
                                        borderValue=(255, 255, 255))
            img = Image.fromarray(img_np)

        return img

    def generate_test_image(
        self,
        puzzle: np.ndarray,
        filename: str,
        font_path: str = None,
        quality: str = "high",
        rotation: float = 0,
        perspective: bool = False
    ):
        """Generate a single test image with specified parameters."""

        # Render base image
        img = self.render_sudoku_image(puzzle, font_path, quality=quality)

        # Apply transformations
        img = self.apply_transformations(img, rotation, perspective)

        # Apply degradation
        img = self.apply_degradation(img, quality)

        # Save
        output_path = os.path.join(self.output_dir, filename)
        img.save(output_path)

        return output_path


def generate_full_test_dataset(num_images=15):
    """Generate a complete test dataset with various configurations."""

    generator = SudokuImageGenerator()

    dataset = {
        "images": []
    }

    configurations = [
        # High quality images
        {"quality": "high", "rotation": 0, "perspective": False, "difficulty": "easy"},
        {"quality": "high", "rotation": 0, "perspective": False, "difficulty": "medium"},
        {"quality": "high", "rotation": 0, "perspective": False, "difficulty": "hard"},

        # Medium quality with variations
        {"quality": "medium", "rotation": 2, "perspective": False, "difficulty": "easy"},
        {"quality": "medium", "rotation": -2, "perspective": False, "difficulty": "medium"},
        {"quality": "medium", "rotation": 0, "perspective": True, "difficulty": "hard"},

        # Low quality (challenging)
        {"quality": "low", "rotation": 3, "perspective": False, "difficulty": "easy"},
        {"quality": "low", "rotation": -3, "perspective": False, "difficulty": "medium"},
        {"quality": "low", "rotation": 0, "perspective": True, "difficulty": "hard"},

        # Mixed variations
        {"quality": "high", "rotation": 1, "perspective": True, "difficulty": "medium"},
        {"quality": "medium", "rotation": -1, "perspective": True, "difficulty": "easy"},
        {"quality": "low", "rotation": 2, "perspective": True, "difficulty": "hard"},

        # Additional variations
        {"quality": "high", "rotation": -1, "perspective": False, "difficulty": "easy"},
        {"quality": "medium", "rotation": 1, "perspective": False, "difficulty": "hard"},
        {"quality": "low", "rotation": 0, "perspective": False, "difficulty": "medium"},
    ]

    # Ensure we have enough configurations
    while len(configurations) < num_images:
        configurations.append({
            "quality": random.choice(["high", "medium", "low"]),
            "rotation": random.uniform(-3, 3),
            "perspective": random.choice([True, False]),
            "difficulty": random.choice(["easy", "medium", "hard"])
        })

    print(f"Generating {num_images} test images...")
    print("=" * 60)

    for i, config in enumerate(configurations[:num_images]):
        # Generate puzzle
        puzzle = generator.generate_puzzle(config["difficulty"])

        # Select font (rotate through available fonts)
        font_idx = i % len(generator.fonts)
        font_path = generator.fonts[font_idx]
        font_name = os.path.basename(font_path) if font_path else "default"

        # Generate filename
        filename = f"test_{i+1:02d}_{config['quality']}_{config['difficulty']}.png"

        print(f"[{i+1}/{num_images}] Generating {filename}")
        print(f"  Font: {font_name}")
        print(f"  Quality: {config['quality']}")
        print(f"  Rotation: {config['rotation']:.1f}°")
        print(f"  Perspective: {config['perspective']}")
        print(f"  Difficulty: {config['difficulty']}")

        # Generate image
        output_path = generator.generate_test_image(
            puzzle=puzzle,
            filename=filename,
            font_path=font_path,
            quality=config['quality'],
            rotation=config['rotation'],
            perspective=config['perspective']
        )

        # Add to dataset
        dataset["images"].append({
            "filename": filename,
            "path": output_path,
            "ground_truth": puzzle.tolist(),
            "config": config,
            "font": font_name
        })

        print(f"  ✓ Saved to {output_path}\n")

    # Add the original testplaatje.png to dataset
    print(f"[{num_images+1}] Adding testplaatje.png (original test image)")
    original_ground_truth = [
        [0, 0, 0, 0, 0, 0, 9, 6, 5],
        [0, 0, 0, 1, 9, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0, 8],
        [1, 0, 0, 7, 6, 0, 0, 0, 0],
        [0, 9, 5, 0, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 1, 0, 5, 3, 0],
        [0, 0, 3, 0, 2, 1, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 1, 5, 0],
        [6, 0, 0, 9, 0, 0, 8, 0, 0]
    ]

    dataset["images"].append({
        "filename": "testplaatje.png",
        "path": "testplaatje.png",
        "ground_truth": original_ground_truth,
        "config": {"quality": "photo", "rotation": 0, "perspective": False},
        "font": "photo"
    })

    # Save dataset JSON
    with open('test_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ Generated {len(dataset['images'])} test images")
    print(f"✓ Dataset saved to test_dataset.json")
    print("=" * 60)

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Sudoku test dataset")
    parser.add_argument(
        "-n", "--num-images",
        type=int,
        default=15,
        help="Number of synthetic images to generate (default: 15)"
    )

    args = parser.parse_args()

    dataset = generate_full_test_dataset(args.num_images)

    print("\nYou can now run the test suite with:")
    print("  python test_ocr_suite.py")
