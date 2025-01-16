import time
import random
import streamlit as st
from colorama import Fore, Back, Style


def show_text(st_ph: st, text: str, position: str, font_size=3):
    """显示字符串

    text：字符串
    position：字符串的位置

    """
    assert position in ["center", "right", "left"]

    st_ph.markdown(
        f"<div align='{position}'><font size={font_size}>{text}</font></div>",
        unsafe_allow_html=True,
    )


def print_colorful(
    *text,
    text_color=None,
    time_color=Back.GREEN + Fore.RED,
    sep: str | None = " ",
    end: str | None = "\n",
    file=None,
    flush: bool = False,
):
    timestamp = time.strftime("%y/%m/%d %H:%M:%S") + " : "
    text = sep.join(list(map(str, text)))
    text = text_color + text + Style.RESET_ALL if text_color is not None else text
    print(
        f"{time_color + timestamp + Style.RESET_ALL}{text}",
        end=end,
        file=file,
        flush=flush,
    )


def random_icon(idx=None):
    icons = "🍇🍈🍉🍊🍋🍌🍍🥭🍎🍏🍐🍑🍒🍓"  # 🐭🐁🐀🐹🐰
    n = len(icons)
    if idx is None:
        return random.sample(icons, 1)[0]
    else:
        return icons[idx % n]
