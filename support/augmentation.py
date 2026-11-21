"""Training-only pitch transposition with an independent, checkpointed RNG."""

import torch


class PitchAugmenter:
    def __init__(self, vocabulary, max_semitones=0, seed=44):
        if not isinstance(max_semitones, int) or not 0 <= max_semitones <= 12:
            raise ValueError("pitch_shift must be an integer from 0 to 12 semitones")
        self.max_semitones = max_semitones
        self.generator = torch.Generator().manual_seed(seed)
        token_to_id = {token: i for i, token in enumerate(vocabulary)}
        self.pitches = torch.tensor([
            int(token.rsplit("_", 1)[1]) if token.startswith(("NOTE_ON_", "NOTE_OFF_")) else -1
            for token in vocabulary
        ])
        maps = []
        for shift in range(-max_semitones, max_semitones + 1):
            mapping = []
            for i, token in enumerate(vocabulary):
                pitch = int(self.pitches[i])
                if pitch < 0:
                    mapping.append(i)
                elif 0 <= pitch + shift <= 127:
                    mapping.append(token_to_id[f"{token.rsplit('_', 1)[0]}_{pitch + shift}"])
                else:
                    mapping.append(-1)  # Never selected: bounds are checked per full window.
            maps.append(mapping)
        self.maps = torch.tensor(maps)

    def __call__(self, x, y):
        """Transpose each aligned CPU x/y pair as one context+1-token window.

        Draw uniformly over allowed integer shifts, including zero. Restrict the
        range using every pitch in x AND y; never clamp individual notes, which
        could change intervals or break note-on/off pairing. Inputs are untouched.
        """
        if self.max_semitones == 0:
            return x, y, torch.zeros(x.shape[0], dtype=torch.long)
        if x.device.type != "cpu" or y.device.type != "cpu":
            raise ValueError("Apply pitch augmentation before moving batches to the device")
        if x.shape != y.shape or x.ndim != 2 or not torch.equal(x[:, 1:], y[:, :-1]):
            raise ValueError("Pitch augmentation requires aligned next-token x/y batches")
        windows = torch.cat((x, y[:, -1:]), dim=1)
        result, shifts = [], []
        for window in windows:
            pitches = self.pitches[window]
            pitches = pitches[pitches >= 0]
            shift = 0
            if pitches.numel():
                low = max(-self.max_semitones, -int(pitches.min()))
                high = min(self.max_semitones, 127 - int(pitches.max()))
                shift = int(torch.randint(low, high + 1, (1,), generator=self.generator))
            result.append(self.maps[shift + self.max_semitones, window])
            shifts.append(shift)
        augmented = torch.stack(result)
        return augmented[:, :-1], augmented[:, 1:], torch.tensor(shifts)
