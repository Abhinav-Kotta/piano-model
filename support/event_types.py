"""Stable event families without changing the token vocabulary or targets."""

EVENT_TYPES = ("BOS", "EOS", "PAD", "NOTE_ON", "NOTE_OFF", "VELOCITY", "TIME_SHIFT")


def event_type_ids(vocabulary):
    return [EVENT_TYPES.index(token if token in EVENT_TYPES else token.rsplit("_", 1)[0])
            for token in vocabulary]
