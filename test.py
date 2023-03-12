import re
from typing import Pattern


def special_chars_pattern() -> Pattern:
    """Returns regex to identify unneeded special chars"""
    return re.compile("[!@#$%^&*()-+{}\[\]\\/<>~`'\"]", flags=re.ASCII)
# [!@#$%^&*()-+{}\[\]\\/<>~`'\"]


patt = special_chars_pattern()
print(patt.sub("", "asd*&(@&@*#&@asd%%"))
# print(patt)
# print(re.sub(r"[!@#$%^&*()-+{}\[\]\\/<>~`'\"]", "", "asd*&(@&@*#&@asd%%"))
# print(patt.fullmatch("*asd"))
# match = patt.fullmatch("abcd*aiudsbsaiud")
# print(match)
