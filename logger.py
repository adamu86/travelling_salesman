from datetime import datetime

def log(msg, level="info"):
    COLORS = {
        "debug": "\033[94m",    # niebieski
        "info": "\033[92m",     # zielony
        "warning": "\033[93m",  # żółty
        "error": "\033[91m",    # czerwony
        "critical": "\033[95m"  # fioletowy
    }

    RESET = "\033[0m"

    level = level.lower()
    color = COLORS.get(level, RESET)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"{color}[{timestamp}] {level.upper():<8} {msg}{RESET}")
