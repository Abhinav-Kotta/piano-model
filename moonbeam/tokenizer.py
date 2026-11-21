"""MIDI <-> complete-note tensors, with tempo-aware elapsed-time quantization."""
from pathlib import Path

import torch
from symusic import Note, Score, Tempo, Track

from .model import BOS, EOS, PAD, MoonbeamConfig, validate_notes


class NoteTokenizer:
    def __init__(self, config=None):
        self.config = config or MoonbeamConfig()

    def encode(self, score, *, include_eos=True):
        """Return [BOS, notes..., EOS] as int64 [length, 6].

        Columns: absolute onset, duration, MIDI pitch//12, MIDI pitch%12,
        program (128 for drums), velocity. Time values are time_step_ms units.
        Initial silence is removed. Pedal/controllers are not represented.
        """
        seconds = score.to('second')
        raw = sorted((float(n.time), int(n.pitch), float(n.duration), int(n.velocity),
                      128 if track.is_drum else int(track.program))
                     for track in seconds.tracks for n in track.notes)
        if not raw:
            raise ValueError('MIDI contains no notes')
        origin = raw[0][0]
        units_per_second = 1000 / self.config.time_step_ms
        rows, previous = [[BOS] * 6], 0
        for onset, pitch, duration, velocity, instrument in raw:
            start = int((onset - origin) * units_per_second + .5)
            end = int((onset + duration - origin) * units_per_second + .5)
            length = max(1, end - start)
            if start - previous > self.config.max_delta:
                raise ValueError(f'Gap of {start - previous} units exceeds max_delta={self.config.max_delta}')
            if length > self.config.max_duration:
                raise ValueError(f'Duration of {length} units exceeds max_duration={self.config.max_duration}')
            rows.append([start, length, pitch // 12, pitch % 12, instrument, velocity])
            previous = start
        if include_eos:
            rows.append([EOS] * 6)
        result = torch.tensor(rows, dtype=torch.long)
        validate_notes(result[None], self.config)
        return result

    def load_midi(self, path, *, include_eos=True):
        return self.encode(Score.from_midi(Path(path).read_bytes()), include_eos=include_eos)

    def decode(self, notes):
        """Render elapsed times at fixed 120 BPM, with one tick per millisecond."""
        notes = notes.detach().cpu()
        validate_notes(notes[None], self.config)
        score = Score(500)
        score.tempos = [Tempo(time=0, qpm=120)]
        tracks = {}
        for onset, duration, octave, pitch_class, instrument, velocity in notes.tolist():
            if onset < 0:
                continue
            if instrument not in tracks:
                track = Track(program=0 if instrument == 128 else instrument, is_drum=instrument == 128)
                tracks[instrument] = track
                score.tracks.append(track)
            tracks[instrument].notes.append(Note(time=onset * self.config.time_step_ms,
                                                 duration=duration * self.config.time_step_ms,
                                                 pitch=octave * 12 + pitch_class, velocity=velocity))
        return score

    def batch(self, sequences):
        """Right-pad independent pieces/windows; never concatenate different pieces."""
        if not sequences or any(s.ndim != 2 or s.shape[-1] != 6 or not len(s) for s in sequences):
            raise ValueError('Provide nonempty [length, 6] tensors')
        if any(s.dtype != torch.long or s.device != sequences[0].device for s in sequences):
            raise ValueError('All sequences must be int64 on the same device')
        if max(map(len, sequences)) > self.config.context_length:
            raise ValueError('Split long sequences into windows no larger than context_length')
        result = sequences[0].new_full((len(sequences), max(map(len, sequences)), 6), PAD)
        for i, sequence in enumerate(sequences):
            result[i, :len(sequence)] = sequence
        validate_notes(result, self.config)
        return result
