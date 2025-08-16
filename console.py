from rich import print as rich_print
import os
import sys
import regex as re

class Console:
    is_debug = False

    @staticmethod
    def set_stream(stream):
        Console.__stream = stream
        
    def ___flush():
        if Console.__stream:
            Console.__stream.flush()
        
    @staticmethod
    def custom(emoji, tag, color, text):
        if Console.__stream:
            text = text.replace("\n", " ")
            # from `text` remove everything between `[` and `]` with regex
            result = re.sub(r'\[.*?\]', '', text).strip()
            # remove multiple spaces
            result = re.sub(r'\s+', ' ', result)
            Console.__stream.write(f"{emoji} [  {tag.upper()}  ] {result}\n")
            Console.___flush()
        rich_print(f"{emoji} [{color} bold][  {tag.upper()}  ][/{color} bold] {text}")
    @staticmethod
    def sam2(text):
        Console.custom("📷", "SAM2", "blue", text)
    @staticmethod
    def clip(text):
        Console.custom("💬", "CLIP", "cyan", text)
    @staticmethod
    def info(text):
        Console.custom("ℹ️", "INFO", "white", text)
    @staticmethod
    def error(text):
        Console.custom("❌", "ERROR", "dark_red", text)
    @staticmethod
    def warning(text):
        Console.custom("⚠️", "WARNING", "yellow", text)
    @staticmethod
    def success(text):
        Console.custom("✅", "SUCCESS", "green", text)
    @staticmethod
    def debug(text):
        if not Console.is_debug: return
        Console.custom("🪳", "DEBUG", "magenta", text)
    @staticmethod
    def raw(message = ""):
        rich_print(message)