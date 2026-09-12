# -*- coding: utf-8 -*-
import re
from typing import Optional, Tuple

_STRIP_CHARS = ' \'\"\u201c\u201d\u2018\u2019'

def strip_quotes(s: str) -> str:
    return s.strip(_STRIP_CHARS)

def has_cjk(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def extract_context_window(full_text: str, target: str, replacement: str, window_chars: int = 4) -> Tuple[str, str]:
    if not full_text or target not in full_text:
        return target, replacement
    idx = full_text.find(target)
    start = max(0, idx - window_chars)
    end = min(len(full_text), idx + len(target) + window_chars)
    prefix = full_text[start:idx].lstrip('\uff0c\u3002\uff01\uff1f,.!?;: ')
    suffix = full_text[idx + len(target):end].rstrip('\uff0c\u3002\uff01\uff1f,.!?;: ')
    orig_window = f"{prefix}{target}{suffix}"
    new_window = f"{prefix}{replacement}{suffix}"
    return orig_window, new_window

def adapt_edit_instruction(
    instruction: str,
    spoken_text: Optional[str] = None,
    is_vocal: bool = False,
) -> str:
    if not instruction:
        return instruction

    lowered = instruction.lower().strip()

    # 1. Add / Insert
    if lowered.startswith('add ') or lowered.startswith('insert '):
        m = re.search(r'^(?:add|insert)\s+(.+?)\s+(after|before)\s+(.+)$', instruction, flags=re.IGNORECASE)
        if m:
            content = strip_quotes(m.group(1))
            pos = m.group(2).lower()
            anchor = strip_quotes(m.group(3))
            if has_cjk(content) or has_cjk(anchor):
                dir_str = '后面' if pos == 'after' else '前面'
                return f'在‘{anchor}’{dir_str}加上‘{content}’'
            else:
                return f"Insert '{content}' {pos} '{anchor}'."

    # 2. Delete / Remove
    if lowered.startswith('delete ') or lowered.startswith('remove '):
        m_anchor = re.search(r'^(?:delete|remove)\s+(.+?)\s+(after|before)\s+(.+)$', instruction, flags=re.IGNORECASE)
        if m_anchor:
            content = strip_quotes(m_anchor.group(1))
            pos = m_anchor.group(2).lower()
            anchor = strip_quotes(m_anchor.group(3))
            if has_cjk(content) or has_cjk(anchor):
                dir_str = '后面' if pos == 'after' else '前面'
                return f'删掉‘{anchor}’{dir_str}的‘{content}’'
            else:
                return f"Remove '{content}' {pos} '{anchor}'."
        m_simple = re.search(r'^(?:delete|remove)\s+(.+)$', instruction, flags=re.IGNORECASE)
        if m_simple:
            content = strip_quotes(m_simple.group(1))
            if has_cjk(content):
                return f'删掉‘{content}’'
            else:
                return f"Remove '{content}'."

    # 3. Replace / Change
    if 'replace ' in lowered or 'change ' in lowered:
        parts = re.split(r'\s+(?:with|to)\s+', instruction, flags=re.IGNORECASE)
        if len(parts) == 2:
            left = re.sub(r'^(?:replace|change)\s+', '', parts[0], flags=re.IGNORECASE)
            right = re.sub(r'\s+in\s+.*$', '', parts[1], flags=re.IGNORECASE)
            orig_word = strip_quotes(left)
            new_word = strip_quotes(right)
            if orig_word and new_word:
                is_lyric = is_vocal or 'lyric' in lowered or 'vocal' in lowered or 'sing' in lowered
                if is_lyric:
                    if spoken_text and orig_word in spoken_text:
                        orig_win, new_win = extract_context_window(spoken_text, orig_word, new_word)
                    else:
                        orig_win, new_win = orig_word, new_word
                    return f'把歌词中的“{orig_win}”改成“{new_win}”。'
                elif has_cjk(orig_word) or has_cjk(new_word):
                    return f'把‘{orig_word}’改成‘{new_word}’'
                else:
                    return f"Replace '{orig_word}' with '{new_word}'."

    if '把' in instruction and ('改成' in instruction or '替换为' in instruction):
        m = re.search(r'把(?:这段)?(?:歌词|说话)?(?:中)?(?:的)?(.+?)(?:改成|替换为|换为)(.+)', instruction)
        if m:
            orig_word = strip_quotes(m.group(1))
            new_word = strip_quotes(m.group(2))
            if orig_word and new_word:
                is_lyric = is_vocal or '歌词' in instruction or '唱' in instruction
                if is_lyric:
                    if spoken_text and orig_word in spoken_text:
                        orig_win, new_win = extract_context_window(spoken_text, orig_word, new_word)
                    else:
                        orig_win, new_win = orig_word, new_word
                    return f'把歌词中的“{orig_win}”改成“{new_win}”。'
                else:
                    return f'把‘{orig_word}’改成‘{new_word}’'

    return instruction
