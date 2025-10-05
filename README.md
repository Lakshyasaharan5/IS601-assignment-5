# IS601-assignment-5
Enhanced Calculator Application with Advanced Design Patterns and pandas 

### Start calculator

```bash
Calculator started. Type 'help' for commands.

Enter command: help

Available commands:
  add, subtract, multiply, divide, power, root - Perform calculations
  history - Show calculation history
  clear - Clear calculation history
  undo - Undo the last calculation
  redo - Redo the last undone calculation
  save - Save calculation history to file
  load - Load calculation history from file
  exit - Exit the calculator

Enter command: 
```

### Add operation

```bash
Enter command: add

Enter numbers (or 'cancel' to abort):
First number: 2
Second number: 4

Result: 6
```

### Subtract operation

```bash
Enter command: subtract

Enter numbers (or 'cancel' to abort):
First number: 5
Second number: 2

Result: 3
```

### Multiply operation

```bash
Enter command: multiply

Enter numbers (or 'cancel' to abort):
First number: 2
Second number: 4

Result: 8
```

### Divide operation

```bash
Enter command: divide

Enter numbers (or 'cancel' to abort):
First number: 6
Second number: 3

Result: 2
```

### Power operation

```bash
Enter command: power

Enter numbers (or 'cancel' to abort):
First number: 3
Second number: 2

Result: 9
```

### Root operation

```bash
Enter command: root

Enter numbers (or 'cancel' to abort):
First number: 16
Second number: 2

Result: 4
```

### History

```bash
Enter command: history

Calculation History:
1. Addition(2, 4) = 6
2. Subtraction(5, 2) = 3
3. Multiplication(1, 4) = 4
4. Multiplication(2, 4) = 8
5. Division(6, 3) = 2
6. Power(3, 2) = 9
7. Root(16, 2) = 4
```

### Undo/Redo operation

```bash
Enter command: undo
Operation undone

Enter command: redo
Operation redone
```

### Save history

```bash
Enter command: save
History saved successfully
```

### Load history

```bash
Enter command: load
History loaded successfully

Enter command: history

Calculation History:
1. Addition(2, 4) = 6
2. Subtraction(5, 2) = 3
3. Multiplication(1, 4) = 4
4. Multiplication(2, 4) = 8
5. Division(6, 3) = 2
6. Power(3, 2) = 9
7. Root(16, 2) = 4
```

### Clear history

```bash
Enter command: clear
History cleared

Enter command: history
No calculations in history
```

### Exit 

```bash
Enter command: exit
History saved successfully.
Goodbye!
```

## Test coverage

```bash
coverage report --fail-under=100
Name                        Stmts   Miss  Cover
-----------------------------------------------
app/__init__.py                 0      0   100%
app/calculation.py             43      0   100%
app/calculator.py             132      0   100%
app/calculator_config.py       42      0   100%
app/calculator_memento.py      11      0   100%
app/calculator_repl.py        106      0   100%
app/exceptions.py               8      0   100%
app/history.py                 23      0   100%
app/input_validators.py        18      0   100%
app/operations.py              62      0   100%
-----------------------------------------------
TOTAL                         445      0   100%
```

## Setup

Clone the repository

```bash
git clone git@github.com:Lakshyasaharan5/IS601-assignment-5.git
```

Create python virtual environment and activate it

```bash
python -m venv venv
source venv/bin/activate
```
Install from requirements.txt

```bash
pip install -r requirements.txt
```

Run pytest and verify

```bash
pytest

============================================== test session starts ===============================================
platform darwin -- Python 3.13.3, pytest-8.3.3, pluggy-1.5.0
rootdir: /Users/lakshyasaharan/projects/IS601/module-5/IS601-assignment-5
plugins: cov-6.0.0, pylint-0.21.0
collected 131 items                                                                                              

tests/test_calculation.py ................                                                                 [ 12%]
tests/test_calculator.py ................................................                                  [ 48%]
tests/test_config.py ..................                                                                    [ 62%]
tests/test_exceptions.py .......                                                                           [ 67%]
tests/test_history.py .......                                                                              [ 73%]
tests/test_operations.py .................                                                                 [ 86%]
tests/test_validators.py ..................                                                                [100%]

============================================== 131 passed in 0.23s ===============================================
```

Run the calculator

```bash
python main.py
```