# Lazy imports — avoid crashing if optional hardware libs (pyaudio, webrtcvad) are absent

def __getattr__(name):
    import importlib
    _map = {
        "AudioEngine": ("src.audio_engine", "AudioEngine"),
        "AudioFrame": ("src.audio_engine", "AudioFrame"),
        "SpeechSegment": ("src.audio_engine", "SpeechSegment"),
        "BargeInEvent": ("src.audio_engine", "BargeInEvent"),
        "FeatureExtractor": ("src.feature_extractor", "FeatureExtractor"),
        "SpeechFeatures": ("src.feature_extractor", "SpeechFeatures"),
        "EmotionClassifier": ("src.emotion_classifier", "EmotionClassifier"),
        "EmotionResult": ("src.emotion_classifier", "EmotionResult"),
        "Emotion": ("src.emotion_classifier", "Emotion"),
        "EMOTION_COLORS": ("src.emotion_classifier", "EMOTION_COLORS"),
        "EMOTION_EMOJIS": ("src.emotion_classifier", "EMOTION_EMOJIS"),
        "SessionManager": ("src.session_manager", "SessionManager"),
        "SessionStats": ("src.session_manager", "SessionStats"),
    }
    if name in _map:
        mod_name, attr = _map[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
