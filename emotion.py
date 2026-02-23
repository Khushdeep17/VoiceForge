from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def detect_emotion(text: str):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    # granular emotion mapping
    if compound >= 0.7:
        emotion = "excited"
    elif 0.4 <= compound < 0.7:
        emotion = "happy"
    elif -0.4 < compound < 0.4:
        emotion = "neutral"
    elif -0.7 < compound <= -0.4:
        emotion = "concerned"
    else:
        emotion = "frustrated"

    return emotion, compound