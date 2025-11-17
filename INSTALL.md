# Sudoku Solver - Installation & Usage

## Installation

Het wrapper script `ssolver.sh` is geïnstalleerd in `~/bin/` en kan vanuit elke directory aangeroepen worden.

### Verificatie

Test of het script werkt:
```bash
~/bin/ssolver.sh --help
```

### Optioneel: Alias toevoegen

Voor nog makkelijker gebruik, voeg een alias toe aan je shell config:

**Voor bash** (`~/.bashrc` of `~/.bash_profile`):
```bash
alias ssolver='~/bin/ssolver.sh'
```

**Voor zsh** (`~/.zshrc`):
```bash
alias ssolver='~/bin/ssolver.sh'
```

Na toevoegen, reload je config:
```bash
source ~/.zshrc  # of source ~/.bashrc
```

Dan kun je simpelweg gebruiken:
```bash
ssolver image.png -o solved.png
```

## Usage

### Basic Usage

```bash
# Met volledige path
~/bin/ssolver.sh puzzle.png -o solved.png

# Of met alias (als ingesteld)
ssolver puzzle.png -o solved.png
```

### Vanuit elke directory

Het script werkt vanuit elke directory op je systeem:

```bash
cd ~/Downloads
ssolver sudoku_photo.jpg -o solved.jpg

cd ~/Desktop
ssolver puzzle.png -o result.png --verbose
```

### Opties

```bash
# Basic solve
ssolver puzzle.png -o solved.png

# Use Tesseract OCR
ssolver puzzle.png -o solved.png --tesseract

# Verbose output
ssolver puzzle.png -o solved.png --verbose

# Debug mode (shows processing steps)
ssolver puzzle.png -o solved.png --debug

# Don't collect training data
ssolver puzzle.png -o solved.png --no-collect

# Display result in window
ssolver puzzle.png -o solved.png --show

# Custom CNN model
ssolver puzzle.png -o solved.png -m models/custom_model.h5
```

### Examples

```bash
# Solve image from anywhere
cd ~/Downloads
ssolver sudoku.png -o ~/Desktop/solved.png

# Solve text file
ssolver sudoku.txt -o solved.txt

# Batch process images
for img in *.png; do
    ssolver "$img" -o "solved_$img" --verbose
done

# Batch process text files
for txt in *.txt; do
    ssolver "$txt" -o "solved_$txt"
done

# Use Tesseract for better results on some images
ssolver difficult.png -o solved.png --tesseract --verbose
```

### Text File Format

The solver accepts text files with the following format:

```
016 000 070
000 096 100
204 800 090

700 080 000
030 070 000
000 300 400

000 000 000
000 769 302
001 003 009
```

Where:
- `0` = empty cell
- `1-9` = filled cell
- Spaces separate groups of 3
- Empty lines separate 3x3 blocks

**Output:**
- `solved.txt` - Solution with original digits and `[filled]` digits marked
- `solved_formatted.txt` - Nicely formatted solution with validation info

## Script Details

**Location**: `~/bin/ssolver.sh`
**Project**: `/Users/mikebos/projecten/sudoka_solver`

Het script:
- ✅ Laadt automatisch de virtual environment
- ✅ Voert `main.py` uit met alle argumenten
- ✅ Werkt vanuit elke directory
- ✅ Geeft exit codes correct door
- ✅ Toont foutmeldingen als project niet gevonden wordt

## Troubleshooting

### "Project directory not found"
Het project is verplaatst. Update de `PROJECT_DIR` variabele in `~/bin/ssolver.sh`:
```bash
nano ~/bin/ssolver.sh
# Pas PROJECT_DIR aan naar nieuwe locatie
```

### "Virtual environment not found"
Virtual environment moet aangemaakt worden:
```bash
cd /Users/mikebos/projecten/sudoka_solver
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Permissions issue
Zorg dat het script executable is:
```bash
chmod +x ~/bin/ssolver.sh
```

## Verwijderen

Om de wrapper te verwijderen:
```bash
rm ~/bin/ssolver.sh

# Als je een alias hebt toegevoegd, verwijder die uit ~/.zshrc of ~/.bashrc
```

---

**Tip**: Voeg `~/bin` toe aan je PATH als dat nog niet het geval is:
```bash
# In ~/.zshrc of ~/.bashrc
export PATH="$HOME/bin:$PATH"
```
