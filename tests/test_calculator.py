import datetime
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import Mock, patch, PropertyMock, MagicMock
from decimal import Decimal
from tempfile import TemporaryDirectory
from app.calculator import Calculator
from app.calculator_repl import calculator_repl
from app.calculator_config import CalculatorConfig
from app.exceptions import OperationError, ValidationError
from app.history import LoggingObserver, AutoSaveObserver
from app.operations import OperationFactory

# Fixture to initialize Calculator with a temporary directory for file paths
@pytest.fixture
def calculator():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = CalculatorConfig(base_dir=temp_path)

        # Patch properties to use the temporary directory paths
        with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
             patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file, \
             patch.object(CalculatorConfig, 'history_dir', new_callable=PropertyMock) as mock_history_dir, \
             patch.object(CalculatorConfig, 'history_file', new_callable=PropertyMock) as mock_history_file:
            
            # Set return values to use paths within the temporary directory
            mock_log_dir.return_value = temp_path / "logs"
            mock_log_file.return_value = temp_path / "logs/calculator.log"
            mock_history_dir.return_value = temp_path / "history"
            mock_history_file.return_value = temp_path / "history/calculator_history.csv"
            
            # Return an instance of Calculator with the mocked config
            yield Calculator(config=config)

# Test Calculator Initialization

def test_calculator_initialization(calculator):
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []
    assert calculator.operation_strategy is None

# Test Logging Setup

@patch('app.calculator.logging.info')
def test_logging_setup(logging_info_mock):
    with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
         patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file:
        mock_log_dir.return_value = Path('/tmp/logs')
        mock_log_file.return_value = Path('/tmp/logs/calculator.log')
        
        # Instantiate calculator to trigger logging
        calculator = Calculator(CalculatorConfig())
        logging_info_mock.assert_any_call("Calculator initialized with configuration")

@patch("builtins.print")
@patch("app.calculator.logging.basicConfig", side_effect=Exception("Logging failed"))
def test_setup_logging_exception(mock_basic_config, mock_print):
    with pytest.raises(Exception, match="Logging failed"):
        Calculator(CalculatorConfig())

    mock_print.assert_any_call("Error setting up logging: Logging failed")

# Test Adding and Removing Observers

def test_add_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    assert observer in calculator.observers

def test_remove_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    calculator.remove_observer(observer)
    assert observer not in calculator.observers

# Test Setting Operations

def test_set_operation(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    assert calculator.operation_strategy == operation

# Test Performing Operations

def test_perform_operation_addition(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    result = calculator.perform_operation(2, 3)
    assert result == Decimal('5')

def test_perform_operation_validation_error(calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    with pytest.raises(ValidationError):
        calculator.perform_operation('invalid', 3)

def test_perform_operation_operation_error(calculator):
    with pytest.raises(OperationError, match="No operation set"):
        calculator.perform_operation(2, 3)

# Test Undo/Redo Functionality

def test_undo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    assert calculator.history == []

def test_redo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    calculator.redo()
    assert len(calculator.history) == 1

# Test History Management

@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history(mock_to_csv, calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.save_history()
    mock_to_csv.assert_called_once()

@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history(mock_exists, mock_read_csv, calculator):
    # Mock CSV data to match the expected format in from_dict
    mock_read_csv.return_value = pd.DataFrame({
        'operation': ['Addition'],
        'operand1': ['2'],
        'operand2': ['3'],
        'result': ['5'],
        'timestamp': [datetime.datetime.now().isoformat()]
    })
    
    # Test the load_history functionality
    try:
        calculator.load_history()
        # Verify history length after loading
        assert len(calculator.history) == 1
        # Verify the loaded values
        assert calculator.history[0].operation == "Addition"
        assert calculator.history[0].operand1 == Decimal("2")
        assert calculator.history[0].operand2 == Decimal("3")
        assert calculator.history[0].result == Decimal("5")
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")
        
def test_perform_operation_generic_exception():
    calc = Calculator(CalculatorConfig())
    calc.operation_strategy = MagicMock()

    with patch("app.calculator.InputValidator.validate_number", return_value=Decimal("2")):
        calc.operation_strategy.execute.side_effect = ZeroDivisionError("division by zero")

        with pytest.raises(OperationError, match="Operation failed: division by zero"):
            calc.perform_operation("2", "0")

@patch("app.calculator.pd.DataFrame.to_csv")
@patch("app.calculator.logging.info")
def test_save_history_empty_else_branch(mock_log_info, mock_to_csv):
    calc = Calculator(CalculatorConfig())

    calc.history = []
    calc.save_history()
    mock_to_csv.assert_called_once()
    mock_log_info.assert_any_call("Empty history saved")

@patch("app.calculator.pd.DataFrame.to_csv", side_effect=Exception("Disk write error"))
@patch("app.calculator.logging.error")
def test_save_history_exception_branch(mock_log_error, mock_to_csv):
    calc = Calculator(CalculatorConfig())

    fake_calc = MagicMock()
    fake_calc.operation = "Addition"
    fake_calc.operand1 = 1
    fake_calc.operand2 = 2
    fake_calc.result = 3
    fake_calc.timestamp.isoformat.return_value = "2025-01-01T00:00:00"
    calc.history = [fake_calc]

    with pytest.raises(OperationError, match="Failed to save history: Disk write error"):
        calc.save_history()

    mock_log_error.assert_any_call("Failed to save history: Disk write error")

@patch("app.calculator.Path.exists", return_value=True) 
@patch("app.calculator.logging.info")
@patch("app.calculator.pd.read_csv")
def test_load_history_empty_file(mock_read_csv, mock_log_info, mock_exists):
    calc = Calculator(CalculatorConfig())
    mock_read_csv.return_value = pd.DataFrame()
    calc.load_history()
    mock_log_info.assert_any_call("Loaded empty history file")



@patch("app.calculator.logging.error")
@patch("app.calculator.pd.read_csv", side_effect=Exception("Corrupted CSV"))
@patch("app.calculator.Path.exists", return_value=True)
def test_load_history_exception_branch(mock_exists, mock_read_csv, mock_log_error):
    calc = Calculator(CalculatorConfig())
    with pytest.raises(OperationError, match="Failed to load history: Corrupted CSV"):
        calc.load_history()
    mock_log_error.assert_any_call("Failed to load history: Corrupted CSV")

def test_get_history_dataframe_for_loop_and_return():
    calc = Calculator(CalculatorConfig())

    fake_calc_1 = MagicMock()
    fake_calc_1.operation = "Addition"
    fake_calc_1.operand1 = 2
    fake_calc_1.operand2 = 3
    fake_calc_1.result = 5
    fake_calc_1.timestamp = "2025-01-01T00:00:00"

    fake_calc_2 = MagicMock()
    fake_calc_2.operation = "Subtraction"
    fake_calc_2.operand1 = 10
    fake_calc_2.operand2 = 4
    fake_calc_2.result = 6
    fake_calc_2.timestamp = "2025-01-01T00:10:00"

    calc.history = [fake_calc_1, fake_calc_2]

    df = calc.get_history_dataframe()

    assert isinstance(df, pd.DataFrame)

    assert df.shape == (2, 5)

    assert "Addition" in df["operation"].values
    assert "Subtraction" in df["operation"].values
    assert 5 in df["result"].astype(int).values
    assert 6 in df["result"].astype(int).values

def test_show_history_empty():
    calc = Calculator(CalculatorConfig())
    calc.history = []
    assert calc.show_history() == []


# Test Clearing History

def test_clear_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.clear_history()
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []

# Test REPL Commands (using patches for input/output handling)

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history') as mock_save_history:
        calculator_repl()
        mock_save_history.assert_called_once()
        mock_print.assert_any_call("History saved successfully.")
        mock_print.assert_any_call("Goodbye!")

@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
def test_calculator_repl_help(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nAvailable commands:")

@patch('builtins.input', side_effect=['add', '2', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_addition(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nResult: 5")

@patch("builtins.input", side_effect=["exit"])
@patch("builtins.print")
def test_calculator_repl_exit_with_exception(mock_print, mock_input):
    # Make save_history() raise an exception
    with patch("app.calculator.Calculator.save_history", side_effect=Exception("Disk full")) as mock_save_history:
        calculator_repl()

        # Ensure save_history was called
        mock_save_history.assert_called_once()

        # Check that the warning message was printed
        mock_print.assert_any_call("Warning: Could not save history: Disk full")

        # And we still should see the Goodbye message
        mock_print.assert_any_call("Goodbye!")

@patch("builtins.input", side_effect=["history", "exit"])
@patch("builtins.print")
def test_calculator_repl_history_empty(mock_print, mock_input):
    # Mock show_history() to return an empty list
    with patch("app.calculator.Calculator.show_history", return_value=[]):
        with patch("app.calculator.Calculator.save_history"):  # needed for 'exit'
            calculator_repl()

    mock_print.assert_any_call("No calculations in history")

@patch("builtins.input", side_effect=["history", "exit"])
@patch("builtins.print")
def test_calculator_repl_history_non_empty(mock_print, mock_input):
    # Mock show_history() to return a list of previous calculations
    fake_history = ["1 + 1 = 2", "2 * 3 = 6"]

    with patch("app.calculator.Calculator.show_history", return_value=fake_history):
        with patch("app.calculator.Calculator.save_history"):  # for 'exit'
            calculator_repl()

    mock_print.assert_any_call("\nCalculation History:")

    mock_print.assert_any_call("1. 1 + 1 = 2")
    mock_print.assert_any_call("2. 2 * 3 = 6")

@patch("builtins.input", side_effect=["clear", "exit"])
@patch("builtins.print")
def test_calculator_repl_clear_history(mock_print, mock_input):
    with patch("app.calculator.Calculator.clear_history") as mock_clear, \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_clear.assert_called_once()
    mock_print.assert_any_call("History cleared")

@patch("builtins.input", side_effect=["undo", "exit"])
@patch("builtins.print")
def test_calculator_repl_undo_success(mock_print, mock_input):
    with patch("app.calculator.Calculator.undo", return_value=True), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Operation undone")

@patch("builtins.input", side_effect=["undo", "exit"])
@patch("builtins.print")
def test_calculator_repl_undo_nothing(mock_print, mock_input):
    with patch("app.calculator.Calculator.undo", return_value=False), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Nothing to undo")

@patch("builtins.input", side_effect=["redo", "exit"])
@patch("builtins.print")
def test_calculator_repl_redo_success(mock_print, mock_input):
    with patch("app.calculator.Calculator.redo", return_value=True), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Operation redone")

@patch("builtins.input", side_effect=["redo", "exit"])
@patch("builtins.print")
def test_calculator_repl_redo_nothing(mock_print, mock_input):
    with patch("app.calculator.Calculator.redo", return_value=False), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Nothing to redo")

@patch("builtins.input", side_effect=["save", "exit"])
@patch("builtins.print")
def test_calculator_repl_save_success(mock_print, mock_input):
    with patch("app.calculator.Calculator.save_history"), \
         patch("app.calculator.Calculator.save_history") as mock_save:
        calculator_repl()

    mock_save.assert_called()
    mock_print.assert_any_call("History saved successfully")

@patch("builtins.input", side_effect=["save", "exit"])
@patch("builtins.print")
def test_calculator_repl_save_exception(mock_print, mock_input):
    with patch("app.calculator.Calculator.save_history", side_effect=Exception("Disk error")):
        calculator_repl()

    mock_print.assert_any_call("Error saving history: Disk error")

@patch("builtins.input", side_effect=["load", "exit"])
@patch("builtins.print")
def test_calculator_repl_load_success(mock_print, mock_input):
    with patch("app.calculator.Calculator.load_history"), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("History loaded successfully")

@patch("builtins.input", side_effect=["load", "exit"])
@patch("builtins.print")
def test_calculator_repl_load_exception(mock_print, mock_input):
    with patch("app.calculator.Calculator.load_history", side_effect=Exception("File missing")), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Error loading history: File missing")

@patch("builtins.input", side_effect=["add", "3", "5", "exit"])
@patch("builtins.print")
def test_calculator_repl_add_success(mock_print, mock_input):
    with patch("app.operations.OperationFactory.create_operation") as mock_factory, \
         patch("app.calculator.Calculator.set_operation") as mock_set_op, \
         patch("app.calculator.Calculator.perform_operation", return_value=Decimal("8")), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_factory.assert_called_once_with("add")
    mock_set_op.assert_called_once()
    mock_print.assert_any_call("\nResult: 8")

@patch("builtins.input", side_effect=["add", "cancel", "exit"])
@patch("builtins.print")
def test_calculator_repl_add_cancel_first(mock_print, mock_input):
    with patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Operation cancelled")

@patch("builtins.input", side_effect=["add", "3", "cancel", "exit"])
@patch("builtins.print")
def test_calculator_repl_add_cancel_second(mock_print, mock_input):
    with patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Operation cancelled")

@patch("builtins.input", side_effect=["add", "3", "5", "exit"])
@patch("builtins.print")
def test_calculator_repl_add_known_exception(mock_print, mock_input):
    with patch("app.operations.OperationFactory.create_operation"), \
         patch("app.calculator.Calculator.set_operation"), \
         patch("app.calculator.Calculator.perform_operation", side_effect=ValidationError("Invalid input")), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Error: Invalid input")

@patch("builtins.input", side_effect=["add", "3", "5", "exit"])
@patch("builtins.print")
def test_calculator_repl_add_unexpected_exception(mock_print, mock_input):
    with patch("app.operations.OperationFactory.create_operation"), \
         patch("app.calculator.Calculator.set_operation"), \
         patch("app.calculator.Calculator.perform_operation", side_effect=Exception("Boom!")), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Unexpected error: Boom!")

@patch("builtins.input", side_effect=["foobar", "exit"])
@patch("builtins.print")
def test_calculator_repl_unknown_command(mock_print, mock_input):
    with patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Unknown command: 'foobar'. Type 'help' for available commands.")

@patch("builtins.input", side_effect=KeyboardInterrupt)
@patch("builtins.print")
def test_calculator_repl_keyboard_interrupt(mock_print, mock_input):
    with patch("app.calculator.Calculator.save_history"):
        # Use side_effect [KeyboardInterrupt, 'exit'] so we exit after handling
        mock_input.side_effect = [KeyboardInterrupt, "exit"]
        calculator_repl()

    mock_print.assert_any_call("\nOperation cancelled")

@patch("builtins.input", side_effect=EOFError)
@patch("builtins.print")
def test_calculator_repl_eof(mock_print, mock_input):
    with patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("\nInput terminated. Exiting...")

@patch("builtins.input", side_effect=[Exception("Input exploded"), "exit"])
@patch("builtins.print")
def test_calculator_repl_inner_exception(mock_print, mock_input):
    with patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    mock_print.assert_any_call("Error: Input exploded")

@patch("logging.error")
@patch("builtins.print")
@patch("app.calculator_repl.Calculator", side_effect=Exception("Init fail"))
def test_calculator_repl_fatal_exception(mock_calc, mock_print, mock_log_error):
    with pytest.raises(Exception, match="Init fail"):
        calculator_repl()

    mock_print.assert_any_call("Fatal error: Init fail")
    mock_log_error.assert_called_once_with("Fatal error in calculator REPL: Init fail")