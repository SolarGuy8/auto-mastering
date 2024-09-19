import librosa
import numpy as np
import soundfile as sf
import json

# Загрузка аудиофайлов
beat_path = 'beat.wav'  # путь к биту
voice_path = 'voice.wav'  # путь к вокалу
output_aligned_path = 'aligned_voice.wav'  # путь для сохранения выровненного вокала
output_mix_path = 'final_mix.wav'  # путь для сохранения сведенного трека
metadata_path = 'metadata.json'  # путь для сохранения метаданных

# Загрузка аудиофайлов с исходной частотой дискретизации
beat, sr_beat = librosa.load(beat_path, sr=None)
voice, sr_voice = librosa.load(voice_path, sr=None)

metadata = {
    'original_sr_beat': int(sr_beat),
    'original_sr_voice': int(sr_voice),
    'operations': []
}

# Проверяем, чтобы оба файла имели одинаковую частоту дискретизации
if sr_beat != sr_voice:
    print("Разная частота дискретизации! Пересэмплируем вокал до частоты бита.")
    voice = librosa.resample(y=voice, orig_sr=sr_voice, target_sr=sr_beat)
    metadata['operations'].append({
        'action': 'resample',
        'from_sr': int(sr_voice),
        'to_sr': int(sr_beat)
    })

# Обнаружение онсетов (начало звуков) с правильными параметрами
beat_onsets = librosa.onset.onset_detect(y=beat, sr=sr_beat, units='samples')
voice_onsets = librosa.onset.onset_detect(y=voice, sr=sr_beat, units='samples')

# Рассчитываем сдвиги на основе первых онсетов
if len(voice_onsets) > 0 and len(beat_onsets) > 0:
    onset_diff = beat_onsets[0] - voice_onsets[0]
else:
    onset_diff = 0

# Рассчитываем временной сдвиг в секундах
time_shift = onset_diff / sr_beat
metadata['onset_diff_samples'] = int(onset_diff)
metadata['time_shift_seconds'] = float(time_shift)

# Сдвигаем вокал на основе рассчитанного сдвига
if onset_diff > 0:
    aligned_voice = np.concatenate([np.zeros(onset_diff), voice])
else:
    aligned_voice = voice[-onset_diff:]

aligned_voice = aligned_voice[:len(beat)]  # Обрезаем по длине бита
metadata['final_voice_length_samples'] = int(len(aligned_voice))

# Сохранение выровненной дорожки
sf.write(output_aligned_path, aligned_voice, sr_beat)
metadata['operations'].append({
    'action': 'shift',
    'shift_samples': int(onset_diff),
    'shift_seconds': float(time_shift)
})

# Сведение бита и вокала
final_mix = beat[:len(aligned_voice)] + aligned_voice  # Складываем сигналы

# Нормализация трека (чтобы не было клиппинга)
final_mix = final_mix / np.max(np.abs(final_mix))

# Сохранение сведенной дорожки
sf.write(output_mix_path, final_mix, sr_beat)
metadata['final_mix_length_samples'] = int(len(final_mix))

# Записываем метаданные в JSON файл
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

# Вывод метаданных в консоль
print(json.dumps(metadata, indent=4))

print(f"Сохранено как: {output_mix_path}")
