import functools
import warnings

warnings.simplefilter("once", DeprecationWarning)


def deprecated(replacement=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"Function {func.__name__} is deprecated."
            if replacement:
                message += f" Use {replacement} instead."
            warnings.warn(message, category=DeprecationWarning)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecation(msg: str) -> None:
    """Deprecation warning wrapper."""
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
