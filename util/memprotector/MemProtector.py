import time
import psutil

print("\nActivating Memory Protector...")
print("Success!")
print("\nMonitoring...")

while True:
    if psutil.virtual_memory().percent > 90:
        processes = []
        for proc in psutil.process_iter():
            if proc.name() == 'python.exe':
                print("Memory leak! Terminating program")
                processes.append((proc, proc.memory_percent()))
        sorted(processes, key=lambda x: x[1])[-1][0].kill()
    time.sleep(10)