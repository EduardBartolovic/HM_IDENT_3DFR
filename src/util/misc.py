def colorstr(*inp):
    """Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code

    Typical usage example:

        colorstr("hello world") # defaults to blue and bold
        colorstr("yellow", "hello world")
        colorstr("green", "underline", "hello world")

    Args:
        inp: Tuple consisting of an optional color + mis + str.
    """
    *args, string = (
        inp if len(inp) > 1 else ("blue", "bold", inp[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def bold(text):
    return f"\033[1m{text}\033[0m"


def underscore(text):
    return f"\033[4m{text}\033[0m"


def safe_round(value, digits=2):
    return round(value, digits) if isinstance(value, (int, float)) else 'N/A'


def smart_round(value, rounding_prec=3):
    if isinstance(value, (int, float)):
        if value == 100:
            return int(100)
        elif value > 99:
            return round(value, rounding_prec)
        else:
            return round(value, 2)
    return value
