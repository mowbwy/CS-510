"""
    Class:   CS-510
    Author:  Joseph Alvayero
    Date:    03/26/2026

    Description:  Each function that needs to be completed has a comment at the top with
                  TODO written in it with instructions.

                  Within the function is a section with the comment #TODO where you will
                  insert your code as per the instructions.
"""  
import os
import psutil
import sys
import threading

"""
  Utility functions provided
"""
def printBlankLines(lines: int):
    for _ in range(lines):
        print("")

def printMsg1(num):
    print("Thread 1 cubed: {}".format(num * num * num))

def printMsg2(num):
    print("Thread 2 squared: {}".format(num * num))


"""
   Displays information about a file and disk usage statistics.
"""
def getFileDiskUsageStatistics() -> None:
    print("Getting Disk Statistics")
    file_name = "./projecttwo.txt"

    # File size
    if os.path.exists(file_name):
        file_size = os.path.getsize(file_name)
        print(f"File: {file_name}")
        print(f"File Size: {file_size} bytes")
    else:
        print(f"File '{file_name}' not found.")
        printBlankLines(2)
        return

    # Disk usage for the directory
    disk_usage = psutil.disk_usage(os.path.dirname(file_name))
    print(f"Disk Total: {disk_usage.total} bytes")
    print(f"Disk Used: {disk_usage.used} bytes")
    print(f"Disk Free: {disk_usage.free} bytes")
    print(f"Disk Percent Used: {disk_usage.percent}%")

    printBlankLines(2)


"""
   Retrieves standard and virtual memory statistics.
"""
def getMemoryStatistics() -> None:
    print("Getting Memory Statistics")

    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()

    print(f"Total Memory: {vm.total} bytes")
    print(f"Available Memory: {vm.available} bytes")
    print(f"Used Memory: {vm.used} bytes")
    print(f"Memory Usage Percent: {vm.percent}%")

    print("")
    print("Swap Memory:")
    print(f"Total Swap: {sm.total} bytes")
    print(f"Used Swap: {sm.used} bytes")
    print(f"Free Swap: {sm.free} bytes")
    print(f"Swap Usage Percent: {sm.percent}%")

    printBlankLines(2)


"""
   Retrieves CPU statistics, including process information.
"""
def getCpuStatistics() -> None:
    print("Getting CPU Statistics")

    print(f"CPU Usage Percent: {psutil.cpu_percent(interval=1)}%")

    cpu_times = psutil.cpu_times()
    print(f"User Time: {cpu_times.user}")
    print(f"System Time: {cpu_times.system}")
    print(f"Idle Time: {cpu_times.idle}")

    print("")
    print("Top 5 Processes by CPU Usage:")

    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    processes = sorted(processes, key=lambda p: p['cpu_percent'], reverse=True)

    for p in processes[:5]:
        print(f"PID {p['pid']} - {p['name']} - {p['cpu_percent']}% CPU")

    printBlankLines(2)


"""
   Demonstrates multi-threading with two threads.
"""
def showThreadingExample() -> None:
    print("Demonstrating Threading")

    t1 = threading.Thread(target=printMsg1, args=(3,))
    t2 = threading.Thread(target=printMsg2, args=(4,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Done With Threading!")
    printBlankLines(2)


"""
   Demonstrates system error handling using divide-by-zero.
"""
def showErrorHandling() -> None:
    print("Demonstrating Error Handling")
    try:
        res = 10 / 0  # Intentional divide-by-zero

    except ZeroDivisionError:
        print("You can't divide by zero!")

    except MemoryError:
        print("Memory Error!")

    else:
        print("Result is", res)

    finally:
        print("Execution complete.")

    printBlankLines(2)


"""
   Main function
"""
def main() -> int:
    print("Starting Program")
    print("=============================")

    getFileDiskUsageStatistics()
    getCpuStatistics()
    getMemoryStatistics()
    showThreadingExample()
    showErrorHandling()

    return 0


if __name__ == '__main__':
    sys.exit(main())
