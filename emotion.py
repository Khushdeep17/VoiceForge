from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from advanced_emotion import detect_emotion_transformer


analyzer = SentimentIntensityAnalyzer()


def detect_emotion(text: str, mode: str = "hybrid"):

    vader = analyzer.polarity_scores(text)
    compound = vader["compound"]

    if mode == "basic":
        if compound >= 0.5:
            return "joy", compound, None
        elif compound <= -0.5:
            return "anger", abs(compound), None
        else:
            return "neutral", abs(compound), None

    # hybrid mode (transformer + fallback)
    try:
        emotion, confidence, _, viz_file = detect_emotion_transformer(text)
        return emotion, confidence, viz_file

    except Exception:
        # fallback to VADER if transformer fails
        if compound >= 0.5:
            return "joy", compound, None
        elif compound <= -0.5:
            return "anger", abs(compound), None
        else:
            return "neutral", abs(compound), None