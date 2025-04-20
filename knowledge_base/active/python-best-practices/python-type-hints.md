---
entry_id: python-type-hints
title: Using Type Hints in Python Code
tags: [python, type-hints, best-practice, mypy, static-typing]
created: 2024-04-19
last_modified: 2024-04-19
status: active
---

# Using Type Hints in Python Code

Type hints are a powerful feature in Python that allows developers to specify the expected types of variables, function parameters, and return values. We use type hints extensively to improve code quality, readability, and maintainability.

## Benefits of Type Hints

- **Improved IDE Support**: Type hints enable better code completion, refactoring, and error detection in IDEs like PyCharm and VS Code.
- **Static Type Checking**: Tools like MyPy can catch type-related errors before runtime.
- **Self-Documenting Code**: Type hints serve as documentation, making it easier to understand function signatures and data structures.
- **Safer Refactoring**: When changing code, type hints help identify potential issues across the codebase.

## Type Hinting Guidelines

### 1. Always Type Hint Function Signatures

```python
def calculate_vehicle_price(base_price: float, age_years: int, mileage: int) -> float:
    """Calculate the adjusted vehicle price based on age and mileage."""
    age_factor = 1.0 - (0.05 * min(age_years, 10))
    mileage_factor = 1.0 - (0.0001 * min(mileage, 100000))
    return base_price * age_factor * mileage_factor
```

### 2. Use Type Aliases for Complex Types

```python
from typing import Dict, List, TypedDict, Tuple

# Type alias for vehicle data
VehicleData = Dict[str, any]

# TypedDict for more specific structure
class VehicleDetails(TypedDict):
    reg_number: str
    make: str
    model: str
    year: int
    mileage: int
    features: List[str]
    previous_owners: int

# Type alias for price history
PriceHistory = List[Tuple[str, float]]  # (date, price)

def analyze_vehicle_history(vehicle: VehicleDetails, price_history: PriceHistory) -> float:
    # Implementation...
    return calculated_value
```

### 3. Use Optional for Parameters That May Be None

```python
from typing import Optional

def get_vehicle_by_id(vehicle_id: str, include_history: Optional[bool] = None) -> VehicleDetails:
    """
    Retrieve vehicle details by ID.
    
    Args:
        vehicle_id: The unique identifier of the vehicle
        include_history: Whether to include vehicle history (defaults to system setting if None)
    
    Returns:
        Complete vehicle details
    """
    # Implementation...
    return vehicle_details
```

### 4. Use Union for Multiple Possible Types

```python
from typing import Union

# A function that can accept either a vehicle ID or a vehicle object
def process_vehicle(vehicle: Union[str, VehicleDetails]) -> None:
    if isinstance(vehicle, str):
        vehicle_details = get_vehicle_by_id(vehicle)
    else:
        vehicle_details = vehicle
    
    # Process the vehicle details...
```

### 5. Type Checking with MyPy

We use MyPy for static type checking in our CI pipeline. Configure your project with a `mypy.ini` file:

```ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True

[mypy.plugins.django-stubs]
django_settings_module = "app.settings"
```

## Common Pitfalls to Avoid

1. **Circular Imports**: Use string literals for forward references to avoid circular import issues.
   ```python
   def get_related_vehicles(vehicle: "Vehicle") -> List["Vehicle"]:
       # Implementation...
   ```

2. **Over-Specifying Types**: Sometimes `Any` is appropriate when the type is truly dynamic.

3. **Forgetting Generic Types**: Always specify the type parameters for containers.
   ```python
   # Incorrect
   vehicles: List = []
   
   # Correct
   vehicles: List[VehicleDetails] = []
   ```

## Resources

- [Python Type Checking Guide](https://mypy.readthedocs.io/en/stable/)
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
