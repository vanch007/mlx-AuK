# -*- coding: utf-8 -*-
import re
from typing import Optional, Tuple

_STRIP_CHARS = ' \'"\u201c\u201d\u2018\u2019'

def strip_quotes(s: str) -> str:
    return s.strip(_STRIP_CHARS)

def extract_context_window(full_text: str, target: str, replacement: str, window_chars: int = 4) -> Tuple[str, str]:
    if not full_text or target not in full_text:
        return target, replacement
    idx = full_text.find(target)
    start = max(0, idx - window_chars)
    end = min(len(full_text), idx + len(target) + window_chars)
    prefix = full_text[start:idx].lstrip('\uff0c\u3002\uff01\uff1f,.!?;: ')
    suffix = full_text[idx + len(target):end].rstrip('\uff0c\u3002\uff01\uff1f,.!?;: ')
    orig_window = f'{prefix}{target}{suffix}'
    new_window = f'{prefix}{replacement}{suffix}'
    return orig_window, new_window

def adapt_edit_instruction(
    instruction: str,
    spoken_text: Optional[str] = None,
    is_vocal: bool = False,
) -> str:
    if not instruction:
        return instruction

    lowered = instruction.lower()
    if 'replace ' in lowered or 'change ' in lowered:
        parts = re.split(r'\s+(?:with|to)\s+', instruction, flags=re.IGNORECASE)
        if len(parts) == 2:
            left = re.sub(r'^(?:replace|change)\s+', '', parts[0], flags=re.IGNORECASE)
            right = re.sub(r'\s+in\s+.*$', '', parts[1], flags=re.IGNORECASE)
            orig_word = strip_quotes(left)
            new_word = strip_quotes(right)

            if orig_word and new_word:
                is_lyric = is_vocal or 'lyric' in lowered or 'vocal' in lowered or 'sing' in lowered
                if spoken_text and orig_word in spoken_text:
                    orig_win, new_win = extract_context_window(spoken_text, orig_word, new_word)
                else:
                    orig_win, new_win = orig_word, new_word
                if is_lyric:
                    return f'\u628a\u6b4c\u8bcd\u4e2d\u7684\u201c{orig_win}\u201d\u6539\u6210\u201c{new_win}\u201d\u3002'
                else:
                    return f'Change "{orig_win}" to "{new_win}" in the recording.'

    if '\u628a' in instruction and ('\u6539\u6210' in instruction or '\u66ff\u6362\u4e3a' in instruction):
        m = re.search(r'\u628a(?:\u8fd9\u6bb5)?(?:\u6b4c\u8bcd|\u8bf4\u8bdd)?(?:\u4e2d)?(?:\u7684)?(.+?)(?:\u6539\u6210|\u66ff\u6362\u4e3a|\u6362\u4e30)(.+)', instruction)
        if m:
            orig_word = strip_quotes(m.group(1))
            new_word = strip_quotes(m.group(2))
            if orig_word and new_word:
                is_lyric = is_vocal or '\u6b4c\u8bcd' in instruction or '\u5531' in instruction
                if spoken_text and orig_word in spoken_text:
                    orig_win, new_win = extract_context_window(spoken_text, orig_word, new_word)
                else:
                    orig_win, new_win = orig_word, new_word
                if is_lyric:
                    return f'\u628a\u6b4c\u8bcd\u4e2d\u7684\u201c{orig_win}\u201d\u6539\u6210\u201c{new_win}\u201d\u3002'
                else:
                    return f'\u628a\u201c{orig_win}\u201d\u6539\u6210\u201c{new_win}\u201d\u3002'

    return instruction
