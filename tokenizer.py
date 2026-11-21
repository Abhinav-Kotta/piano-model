from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizerConfig:
    time_step: int = 10
    velocity_bins: int = 32
    max_time_shift: int = 960

    def __post_init__(self):
        if self.time_step < 1 or self.max_time_shift < 1 or self.max_time_shift % self.time_step:
            raise ValueError("time_step must be positive and divide max_time_shift")
        if not 1 <= self.velocity_bins <= 127:
            raise ValueError("velocity_bins must be between 1 and 127")

    def velocity_value(self, velocity):
        if not 1 <= velocity <= 127:
            raise ValueError(f"Invalid note-on velocity: {velocity}")
        index = (velocity - 1) * self.velocity_bins // 127
        return self.velocity_values[index]

    @property
    def velocity_values(self):
        # Representative bin centers; labels remain decodable MIDI velocities.
        return [1 + (2 * i + 1) * 127 // (2 * self.velocity_bins)
                for i in range(self.velocity_bins)]


LEGACY_TOKENIZER = TokenizerConfig(time_step=1, velocity_bins=127)


def preprocess_performance(performance_midi, target_tpq=None, time_step=1):
    """Copy notes and shift the first onset to zero.

    Pass target_tpq for cross-file training. None preserves the original
    single-performance experiment's tick resolution and first-track selection.
    """
    if time_step < 1:
        raise ValueError("time_step must be positive")
    if target_tpq is not None:
        if target_tpq <= 0:
            raise ValueError("target_tpq must be positive")
        score = performance_midi.resample(target_tpq, min_dur=1)
        tracks = [track for track in score.tracks if not track.is_drum]
    else:
        score = performance_midi.copy()
        tracks = score.tracks[:1]
    performance_notes = sorted(
        (note for track in tracks for note in track.notes),
        key=lambda note: (note.time, note.pitch),
    )
    if not performance_notes:
        raise ValueError("Performance contains no notes")

    start_tick = performance_notes[0].time
    for note in performance_notes:
        start = note.time - start_tick
        if time_step == 1:
            note.time = start
            continue
        end = start + note.duration
        # Quantize absolute onsets/endpoints, not successive gaps (which drift).
        quantized_start = (start + time_step // 2) // time_step * time_step
        quantized_end = (end + time_step // 2) // time_step * time_step
        note.time = quantized_start
        note.duration = max(time_step, quantized_end - quantized_start)

    return performance_notes


def create_event(performance_notes):
    event_list = []

    for note in performance_notes:
        event_list.append({
            "time": note.time,
            "type": "note_on",
            "pitch": note.pitch,
            "velocity": note.velocity
        })

        event_list.append({
            "time": note.time + note.duration,
            "type": "note_off",
            "pitch": note.pitch
        })

    event_priority = {"note_off": 0, "note_on": 1}

    event_list.sort(key=lambda event: (event["time"], event_priority[event[
        "type"]], event["pitch"]))

    time_shifted_events = []

    current_time = 0
    for event in event_list:
        gap = event["time"] - current_time

        if gap > 0:
            time_shifted_events.append({"type": "time_shift", "ticks": gap})

        note_event = {}
        for k, v in event.items():
            if k != 'time':
                note_event[k] = v
        time_shifted_events.append(note_event)

        current_time = event["time"]

    return time_shifted_events


def define_vocab(config=None):
    config = config or LEGACY_TOKENIZER
    vocabulary = ['BOS', 'EOS', 'PAD']
    for pitch in range(128):
        vocabulary.append(f"NOTE_ON_{pitch}")
        vocabulary.append(f"NOTE_OFF_{pitch}")

    for velocity in config.velocity_values:
        vocabulary.append(f"VELOCITY_{velocity}")

    for ticks in range(config.time_step, config.max_time_shift + 1, config.time_step):
        vocabulary.append(f"TIME_SHIFT_{ticks}")

    token_to_id, id_to_token = {}, {}

    for i, t in enumerate(vocabulary):
        token_to_id[t] = i
        id_to_token[i] = t

    return vocabulary, token_to_id, id_to_token


def tokenize(token_to_id, time_shifted_events, config=None):
    config = config or LEGACY_TOKENIZER
    tokens = []
    for event in time_shifted_events:
        event_type = event['type']

        if event_type == 'note_on':
            pitch = event['pitch']
            velocity = event['velocity']
            if not 0 <= pitch <= 127:
                raise ValueError(f"Pitch outside vocabulary: {pitch}")

            if not 1 <= velocity <= 127:
                raise ValueError(f"Invalid note-on velocity: {velocity}")

            tokens.append(f"VELOCITY_{config.velocity_value(velocity)}")
            tokens.append(f"NOTE_ON_{pitch}")

        elif event_type == 'note_off':
            pitch = event['pitch']
            if not 0 <= pitch <= 127:
                raise ValueError(f"Pitch outside vocabulary {pitch}")

            tokens.append(f"NOTE_OFF_{pitch}")

        elif event_type == 'time_shift':
            ticks = event['ticks']
            if ticks < 0:
                raise ValueError(f"negative time shift: {ticks}")
            if ticks % config.time_step:
                raise ValueError("Time shifts must be on the configured grid; quantize notes first")

            while ticks > 0:
                shift = min(ticks, config.max_time_shift)
                tokens.append(f"TIME_SHIFT_{shift}")
                ticks -= shift

        else:
            raise ValueError("unknown event type")

    tokens.append('EOS')
    token_ids = []
    for token in tokens:
        token_ids.append(token_to_id[token])

    return tokens, token_ids
