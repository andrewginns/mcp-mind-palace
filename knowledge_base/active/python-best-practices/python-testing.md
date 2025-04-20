---
entry_id: python-testing-practices
title: Testing Best Practices for Python Code
tags: [python, testing, pytest, unit-tests, integration-tests, mocking]
created: 2024-04-19
last_modified: 2024-04-19
status: active
---

# Testing Best Practices for Python Code

Effective testing is a cornerstone of engineering practices. This guide outlines our approach to testing Python code, ensuring reliability and maintainability across our services.

## Testing Philosophy

We follow these testing principles:

1. **Test-Driven Development (TDD)** when appropriate, especially for complex business logic
2. **Comprehensive test coverage** for critical paths and edge cases
3. **Fast and reliable tests** that can run frequently during development
4. **Isolation** between tests to prevent interdependencies

## Test Types and When to Use Them

### Unit Tests

Unit tests verify that individual components work as expected in isolation.

```python
# Example unit test for a pricing calculation function
def test_calculate_vehicle_price():
    # Arrange
    base_price = 10000
    age_years = 5
    mileage = 50000
    
    # Act
    result = calculate_vehicle_price(base_price, age_years, mileage)
    
    # Assert
    expected = 7500  # Based on our pricing formula
    assert result == expected, f"Expected {expected}, got {result}"
```

### Integration Tests

Integration tests verify that components work together correctly.

```python
# Example integration test for vehicle data processing pipeline
def test_vehicle_data_processing_pipeline():
    # Arrange
    test_data = load_test_vehicle_data()
    
    # Act
    processed_data = process_vehicle_data_pipeline(test_data)
    
    # Assert
    assert len(processed_data) == len(test_data)
    assert all(vehicle['processed'] for vehicle in processed_data)
    # Check specific transformations...
```

### End-to-End Tests

E2E tests verify that entire workflows function correctly from a user perspective.

```python
# Example E2E test using pytest-django
@pytest.mark.django_db
def test_vehicle_listing_workflow(client, test_user, test_vehicle):
    # Login
    client.login(username=test_user.username, password='password')
    
    # Create listing
    response = client.post('/api/listings/create/', {
        'vehicle_id': test_vehicle.id,
        'asking_price': 15000,
        'description': 'Test listing'
    })
    assert response.status_code == 201
    
    # Verify listing appears in user's dashboard
    response = client.get('/api/user/listings/')
    assert response.status_code == 200
    assert test_vehicle.id in [listing['vehicle_id'] for listing in response.json()]
```

## Testing Tools

### pytest

We use pytest as our primary testing framework:

```python
# pytest fixture example
@pytest.fixture
def test_vehicle_data():
    return {
        'reg_number': 'AB12CDE',
        'make': 'Tesla',
        'model': 'Model 3',
        'year': 2021,
        'mileage': 15000
    }

def test_vehicle_validation(test_vehicle_data):
    result = validate_vehicle_data(test_vehicle_data)
    assert result.is_valid
    assert not result.errors
```

### Mock and Patch

For isolating components and simulating external dependencies:

```python
from unittest.mock import patch, MagicMock

def test_vehicle_api_client():
    # Mock the external API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'vehicle': {'reg': 'AB12CDE', 'make': 'BMW'}}
    
    with patch('requests.get', return_value=mock_response):
        client = VehicleAPIClient()
        vehicle = client.get_vehicle_by_reg('AB12CDE')
        
        assert vehicle['make'] == 'BMW'
        assert 'reg' in vehicle
```

### Factory Boy

For generating test data:

```python
# Example factory for Vehicle model
class VehicleFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Vehicle
    
    reg_number = factory.Sequence(lambda n: f'TEST{n:04d}')
    make = factory.Faker('company')
    model = factory.Faker('word')
    year = factory.Faker('random_int', min=2000, max=2023)
    mileage = factory.Faker('random_int', min=0, max=100000)
```

## Test Organization

### Directory Structure

```
my_project/
├── tests/
│   ├── unit/
│   │   ├── test_models.py
│   │   └── test_utils.py
│   ├── integration/
│   │   └── test_api.py
│   ├── e2e/
│   │   └── test_workflows.py
│   └── conftest.py  # Shared fixtures
```

### Naming Conventions

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

## CI Integration

All tests run in our CI pipeline:

1. Unit tests run on every commit
2. Integration tests run on PRs and merges to main
3. E2E tests run nightly and before releases

## Best Practices

1. **Keep tests fast**: Optimize slow tests, use appropriate mocking
2. **Test behavior, not implementation**: Focus on what the code does, not how it does it
3. **One assertion per test**: When possible, test one specific behavior per test function
4. **Use descriptive test names**: Names should describe what is being tested
5. **Don't test third-party code**: Focus on testing your own code
6. **Use parameterized tests** for testing multiple inputs:

```python
@pytest.mark.parametrize("mileage,expected_factor", [
    (0, 1.0),
    (50000, 0.95),
    (100000, 0.9),
    (150000, 0.9)  # Capped at 100000
])
def test_mileage_factor(mileage, expected_factor):
    assert calculate_mileage_factor(mileage) == expected_factor
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Factory Boy Documentation](https://factoryboy.readthedocs.io/)
