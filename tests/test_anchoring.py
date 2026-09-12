import sys
sys.path.insert(0, '/Users/vanch/mlx-AuK/src')

from mlx_auk.anchoring import adapt_edit_instruction

def test_vocal_zh_anchoring():
    # vocaledit-zh-2
    instr = 'Replace “恩怨” with “笑容” in the lyrics'
    spoken = '当恩怨搁一半，我怎么圈揽。看灯笼血红染，寻仇已太晚'
    res = adapt_edit_instruction(instr, spoken_text=spoken, is_vocal=True)
    print("vocaledit-zh-2 adapted:", res)
    assert "当恩怨" in res and "当笑容" in res
    assert res.startswith("把歌词中的“")

def test_vocal_zh1_anchoring():
    # vocaledit-zh-1
    instr = 'Replace “寂寞” with “疯狂” in the lyrics'
    spoken = '这是今天最寂寞的时候，太阳照着你好温柔'
    res = adapt_edit_instruction(instr, spoken_text=spoken, is_vocal=True)
    print("vocaledit-zh-1 adapted:", res)
    assert "寂寞" in res and "疯狂" in res

def test_speech_zh_anchoring():
    instr = '把“接受现实”改成“勇敢前行”'
    spoken = '我们需要接受现实，才能继续生活'
    res = adapt_edit_instruction(instr, spoken_text=spoken, is_vocal=False)
    print("speech content edit adapted:", res)
    assert "接受现实" in res and "勇敢前行" in res

if __name__ == '__main__':
    test_vocal_zh_anchoring()
    test_vocal_zh1_anchoring()
    test_speech_zh_anchoring()
    print("✅ All anchoring unit tests passed successfully!")

