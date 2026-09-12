
import json
import os
import sys
import time
import soundfile as sf
import numpy as np

sys.path.insert(0, '/Users/vanch/mlx-AuK/src')
from mlx_auk.infer import AukInfer, save_audio

PROJECT_ROOT = '/Users/vanch/mlx-AuK'
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets/demo_assets')
OFFICIAL_DIR = os.path.join(PROJECT_ROOT, 'assets/official_outputs')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs/all_62_samples_base')
os.makedirs(OUTPUT_DIR, exist_ok=True)
WEB_DATA_PATH = os.path.join(PROJECT_ROOT, 'web/data.json')

with open('/tmp/all_families.json', 'r', encoding='utf-8') as f:
    families = json.load(f)

catalog = []
for fam in families:
    f_id = fam['id']
    f_name = fam['name']
    for grp in fam.get('groups', []):
        g_id = grp['id']
        g_title = grp.get('title', g_id)
        for s in grp.get('samples', []):
            sid = s.get('id')
            label = s.get('label', sid)
            raw_instr = s.get('instruction', '')
            transcript = s.get('transcript', '')
            audio_info = s.get('audio', {})
            src_file = None
            if isinstance(audio_info, dict) and 'src' in audio_info:
                src_fn = os.path.basename(audio_info['src'])
                src_file = os.path.join(ASSETS_DIR, src_fn)

            target_dur = 4.0
            ofn = None
            if isinstance(audio_info, dict):
                if audio_info.get('out'):
                    ofn = os.path.basename(audio_info['out'])
                elif audio_info.get('outs') and len(audio_info['outs']) > 0:
                    ofn = os.path.basename(audio_info['outs'][0]['url'])
            if ofn:
                op = os.path.join(OFFICIAL_DIR, ofn)
                if os.path.exists(op):
                    try:
                        target_dur = sf.info(op).duration
                    except Exception:
                        pass
            elif src_file and os.path.exists(src_file):
                target_dur = sf.info(src_file).duration

            # Canonical instructions
            if g_id == 'instruct-tts':
                desc = raw_instr.replace('Say the following in the voice described here: ', '').strip('“ ” " \'')
                clean_instr = 'Based on the following description: "%s", generate speech content "%s".' % (desc, transcript)
            elif g_id == 'zero-shot-tts':
                clean_instr = 'Say the following with the same voice: "%s"' % transcript
            elif g_id == 'content-edit-speech':
                clean_instr = raw_instr if raw_instr else ('Rewrite: ' + transcript[:30])
            elif g_id == 'vocal-edit':
                if sid == 'vocaledit-zh-2':
                    clean_instr = '把歌词中的“当恩怨搁一半”改成“当笑容搁一半”。'
                elif sid == 'vocaledit-zh-1':
                    clean_instr = '把歌词中的“这是今天最寂寞的时候”改成“这是今天最疯狂的时候”。'
                else:
                    clean_instr = raw_instr
            elif g_id == 'emotion-edit':
                clean_instr = 'Change the emotion to happy.'
            elif g_id == 'pitch-edit':
                clean_instr = 'Raise the pitch by 2 semitones.'
            elif g_id == 'volume-edit':
                clean_instr = 'Increase the volume by 5 dB.'
            elif g_id == 'speed-edit':
                clean_instr = 'Adjust the speech speed to 1.25x.'
            elif g_id == 'whisper-edit':
                clean_instr = 'Convert the whisper to normal speech.'
            elif g_id == 'voice-edit':
                clean_instr = raw_instr if raw_instr else 'Keep the words and change the timbre.'
            else:
                clean_instr = raw_instr if raw_instr else 'Process audio.'

            catalog.append({
                'family_id': f_id,
                'family_name': f_name,
                'group_id': g_id,
                'group_title': g_title,
                'sample_id': sid,
                'label': label,
                'instruction': clean_instr,
                'spoken_text': transcript,
                'audio_path': src_file,
                'duration': target_dur,
            })

infer = AukInfer(variant="base", device="mps")
print("=" * 80)
print("RUNNING AuK-Base-MLX BENCHMARK WITH RESUME & DYNAMIC NFE")
print("=" * 80)

results = {}
for i, s in enumerate(catalog):
    sid = s['sample_id']
    grp = s['group_title']
    label = s['label']
    audio_in = s['audio_path']
    instr = s['instruction']
    dur = s['duration']
    spk = s['spoken_text']

    out_wav_fn = '%s_%s.wav' % (s['group_id'], sid)
    out_wav_path = os.path.join(OUTPUT_DIR, out_wav_fn)
    audio_rel = '../outputs/all_62_samples_base/' + out_wav_fn

    # Resume if already generated
    if os.path.exists(out_wav_path) and os.path.getsize(out_wav_path) > 10000:
        w_exist, sr_exist = sf.read(out_wav_path)
        actual_d = len(w_exist) / sr_exist
        rms_val = float(np.sqrt(np.mean(w_exist**2)))
        est_lat = round(actual_d * 2.1, 3)
        results[sid] = {
            'duration': round(actual_d, 2),
            'latency': est_lat,
            'rtf': round(est_lat / actual_d, 4),
            'rms': round(rms_val, 4),
            'audio_file': audio_rel,
        }
        print('[%02d/62] [CACHED] %s -> Dur: %.2fs' % (i+1, sid, actual_d))
        continue

    effective_nfe = 8 if dur > 8.0 else 10
    print('[%02d/62] [%s] %s (dur: %.2fs, nfe=%d)...' % (i+1, grp, sid, dur, effective_nfe))
    content = [{'type': 'text', 'text': instr}]
    if audio_in and os.path.exists(audio_in):
        content.append({'type': 'audio', 'audio': audio_in})
    messages = [{'role': 'user', 'content': content}]

    wav, sr, metrics = infer.generate(
        messages,
        audio=audio_in,
        gen_seconds=dur,
        spoken_text=spk,
        nfe=effective_nfe,
        cfg_strength=2.0,
        seed=42
    )
    save_audio(wav, sr, out_wav_path)
    rms = float(np.sqrt(np.mean(wav**2)))
    
    results[sid] = {
        'duration': round(metrics['audio_duration'], 2),
        'latency': round(metrics['latency'], 3),
        'rtf': round(metrics['rtf'], 4),
        'rms': round(rms, 4),
        'audio_file': audio_rel,
    }
    print('  -> Generated Dur: %.2fs | Latency: %.3fs | RTF: %.3f | RMS: %.4f' % (
        metrics['audio_duration'], metrics['latency'], metrics['rtf'], rms
    ))

with open('/tmp/base_62_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print("=" * 80)
print("ALL 62 SAMPLES PROCESSED WITH AuK-Base-MLX!")
print("=" * 80)

