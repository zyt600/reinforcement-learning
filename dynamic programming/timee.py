import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to run.")
        return result
    return wrapper

@timing_decorator
def slow_function():
    time.sleep(2)
    print("Function has finished running")

slow_function()
