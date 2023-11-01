import concurrent.futures
import time


def task(n):
    print(f"Executing Task {n}")
    time.sleep(n)
    return f"Task {n} Completed"


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(task, n) for n in range(3, 0, -1)]

        for future in concurrent.futures.as_completed(futures):
            print(future.result())

print("All tasks are completed!")
