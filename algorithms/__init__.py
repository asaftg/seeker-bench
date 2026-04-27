"""Pure-function algorithm extracts.

Each module here is a stateless (or explicitly-state-passing) version
of a piece of logic that lives in a manager. Live code calls the same
function so live + replay share one code path bit-for-bit. New algos
land here as we extract them — track_predictor first, classifier and
fusion candidates next.
"""
