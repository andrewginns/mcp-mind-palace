---
entry_id: python-error-handling
title: Error Handling and Logging in Python
tags: [python, error-handling, exceptions, logging, monitoring]
created: 2024-04-19
last_modified: 2024-04-19
status: active
---

# Error Handling and Logging in Python

Proper error handling and logging are essential for building robust, maintainable Python applications. This guide outlines best practices for handling exceptions and implementing effective logging.

## Exception Handling Principles

We follow these principles for exception handling:

1. **Be specific**: Catch only the exceptions you can handle meaningfully
2. **Fail fast**: Let exceptions propagate when you can't handle them properly
3. **Preserve context**: Always include original exceptions when re-raising
4. **Design for recovery**: Structure code to allow for graceful degradation

## Exception Handling Patterns

### Try-Except-Else-Finally Pattern

```python
def process_vehicle_data(vehicle_id):
    try:
        # Code that might raise an exception
        raw_data = fetch_vehicle_data(vehicle_id)
    except ConnectionError as e:
        # Handle specific exception
        logger.error(f"Connection error fetching vehicle {vehicle_id}: {e}")
        return None
    except Exception as e:
        # Handle unexpected exceptions
        logger.exception(f"Unexpected error processing vehicle {vehicle_id}")
        raise ProcessingError(f"Failed to process vehicle {vehicle_id}") from e
    else:
        # Code that runs if no exception occurs
        processed_data = transform_vehicle_data(raw_data)
        return processed_data
    finally:
        # Code that always runs
        cleanup_resources()
```

### Custom Exception Hierarchy

We maintain a clear exception hierarchy for our applications:

```python
# Base exception for all app specific exceptions
class AppError(Exception):
    """Base class for all exceptions."""
    pass

# Domain-specific exceptions
class VehicleError(AppError):
    """Base class for vehicle-related exceptions."""
    pass

class VehicleNotFoundError(VehicleError):
    """Raised when a vehicle cannot be found."""
    pass

class VehicleDataInvalidError(VehicleError):
    """Raised when vehicle data fails validation."""
    def __init__(self, vehicle_id, validation_errors):
        self.vehicle_id = vehicle_id
        self.validation_errors = validation_errors
        message = f"Invalid data for vehicle {vehicle_id}: {validation_errors}"
        super().__init__(message)
```

### Context Managers for Resource Management

```python
from contextlib import contextmanager

@contextmanager
def vehicle_data_transaction(vehicle_id):
    """Context manager for handling vehicle data transactions."""
    transaction = None
    try:
        transaction = start_transaction(vehicle_id)
        yield transaction
    except Exception as e:
        if transaction:
            rollback_transaction(transaction)
        logger.exception(f"Transaction failed for vehicle {vehicle_id}")
        raise
    else:
        if transaction:
            commit_transaction(transaction)

# Usage
def update_vehicle_price(vehicle_id, new_price):
    with vehicle_data_transaction(vehicle_id) as transaction:
        transaction.update_price(new_price)
        transaction.log_price_change(new_price)
    return True
```

## Logging Best Practices

### Logging Configuration

We use Python's built-in logging module with a standardized configuration:

```python
import logging
import logging.config
import yaml
import os

def setup_logging(
    default_path='logging_config.yaml',
    default_level=logging.INFO,
    env_key='LOG_CONFIG'
):
    """Setup logging configuration"""
    path = os.getenv(env_key, default_path)
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
```

### Logging Levels

Use appropriate logging levels:

- **DEBUG**: Detailed information, typically useful only for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Indication that something unexpected happened, but the application is still working
- **ERROR**: Due to a more serious problem, the application has not been able to perform a function
- **CRITICAL**: A serious error indicating that the application itself may be unable to continue running

```python
def process_vehicle_listing(vehicle_data):
    logger = logging.getLogger(__name__)
    
    logger.debug(f"Processing vehicle data: {vehicle_data}")
    
    if not validate_vehicle_data(vehicle_data):
        logger.warning(f"Invalid vehicle data received: {vehicle_data}")
        return False
    
    try:
        listing_id = create_vehicle_listing(vehicle_data)
        logger.info(f"Created vehicle listing {listing_id} for {vehicle_data['reg_number']}")
        return listing_id
    except DatabaseError as e:
        logger.error(f"Database error creating listing for {vehicle_data['reg_number']}: {e}")
        return False
    except Exception as e:
        logger.critical(f"Unexpected error in listing creation: {e}", exc_info=True)
        raise
```

### Structured Logging

For better searchability and analysis, use structured logging:

```python
def log_vehicle_event(vehicle_id, event_type, details):
    logger = logging.getLogger("vehicle.events")
    logger.info("Vehicle event", extra={
        "vehicle_id": vehicle_id,
        "event_type": event_type,
        "details": details,
        "timestamp": datetime.utcnow().isoformat()
    })
```

## Monitoring and Alerting

### Integration with Monitoring Tools

We integrate our logging with monitoring tools:

```python
def monitor_critical_operation(operation_name):
    """Decorator to monitor critical operations and send alerts on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = logging.getLogger("monitoring")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful execution with timing
                logger.info(f"{operation_name} completed", extra={
                    "operation": operation_name,
                    "execution_time": execution_time,
                    "status": "success"
                })
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log failure with details
                logger.error(f"{operation_name} failed", extra={
                    "operation": operation_name,
                    "execution_time": execution_time,
                    "status": "failure",
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
                # Send alert for critical failures
                send_alert(
                    title=f"Critical operation failed: {operation_name}",
                    message=f"Error: {e}",
                    level="critical"
                )
                
                raise
        return wrapper
    return decorator

# Usage
@monitor_critical_operation("vehicle_import_batch")
def import_vehicle_batch(batch_data):
    # Implementation...
```

## Error Handling in Asynchronous Code

For async code, use appropriate error handling patterns:

```python
async def fetch_vehicle_data_async(vehicle_id):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"/api/vehicles/{vehicle_id}") as response:
                if response.status == 404:
                    raise VehicleNotFoundError(vehicle_id)
                elif response.status != 200:
                    raise VehicleError(f"API error: {response.status}")
                
                return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"Network error fetching vehicle {vehicle_id}: {e}")
        raise VehicleError("Network error") from e
```

## Resources

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Python Exception Handling Best Practices](https://docs.python.org/3/tutorial/errors.html)
