from emotion import detect_emotion

tests = [
    "This is the best news ever!",
    "I am happy with the results.",
    "The meeting is at 5 PM.",
    "I am worried about this issue.",
    "This is completely unacceptable and frustrating."
]

for text in tests:
    emotion, intensity = detect_emotion(text)

    print(f"\nText: {text}")
    print(f"Emotion: {emotion}")
    print(f"Intensity: {intensity:.3f}")