"""
Mood tracker for AI/ML project."""

from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

MOOD_LABELS = ["Happy", "Calm", "Neutral", "Anxious", "Sad", "Angry"]
MOOD_SCORE = {
    "Happy": 5,
    "Calm": 4,
    "Neutral": 3,
    "Anxious": 2,
    "Sad": 1,
    "Angry": 0,
}
MOOD_LOG_FILE = "mood_log.csv"

TRAINING_EXAMPLES = [
    ("I feel cheerful and energetic", "Happy"),
    ("This is a peaceful and relaxed day", "Calm"),
    ("I am okay, nothing special happened", "Neutral"),
    ("I am worried and tense about work", "Anxious"),
    ("I feel sad and down", "Sad"),
    ("I am angry and frustrated", "Angry"),
    ("I am excited and joyful", "Happy"),
    ("I am nervous before the presentation", "Anxious"),
    ("I feel calm and grounded", "Calm"),
    ("I am feeling lonely and upset", "Sad"),
    ("I feel peaceful and content", "Calm"),
    ("I am in a great mood today", "Happy"),
    ("I am annoyed and irritated", "Angry"),
    ("I am anxious about the next meeting", "Anxious"),
    ("I am not sure how I feel", "Neutral"),
    ("I feel relaxed and centered", "Calm"),
    ("Today is a happy day", "Happy"),
    ("I feel gloomy and low", "Sad"),
    ("I am upset and furious", "Angry"),
]


@dataclass
class MoodEntry:
    timestamp: datetime
    description: str
    mood: str
    confidence: float

    def to_csv_row(self) -> List[str]:
        return [
            self.timestamp.isoformat(),
            self.description,
            self.mood,
            f"{self.confidence:.2f}",
        ]

    @staticmethod
    def from_csv_row(row: List[str]) -> MoodEntry:
        timestamp = datetime.fromisoformat(row[0])
        description = row[1]
        mood = row[2]
        confidence = float(row[3])
        return MoodEntry(timestamp, description, mood, confidence)


class MoodClassifier:
    def __init__(self) -> None:
        self.vectorizer = CountVectorizer(lowercase=True, stop_words="english")
        self.model = MultinomialNB()
        self._train_model()

    def _train_model(self) -> None:
        texts, labels = zip(*TRAINING_EXAMPLES)
        matrix = self.vectorizer.fit_transform(texts)
        self.model.fit(matrix, labels)

    def predict(self, text: str) -> Tuple[str, float]:
        vector = self.vectorizer.transform([text])
        label = self.model.predict(vector)[0]
        probability = float(self.model.predict_proba(vector).max())
        return label, probability


class MoodTracker:
    def __init__(self, log_path: str = MOOD_LOG_FILE) -> None:
        self.log_path = log_path
        self.classifier = MoodClassifier()
        self.entries: List[MoodEntry] = []
        self._load_log()

    def _load_log(self) -> None:
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) != 4:
                    continue
                try:
                    entry = MoodEntry.from_csv_row(row)
                except ValueError:
                    continue
                self.entries.append(entry)

    def save_entry(self, entry: MoodEntry) -> None:
        self.entries.append(entry)
        file_exists = os.path.exists(self.log_path)
        with open(self.log_path, "a", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(["timestamp", "description", "mood", "confidence"])
            writer.writerow(entry.to_csv_row())

    def add_entry(self, description: str) -> MoodEntry:
        mood, confidence = self.classifier.predict(description)
        entry = MoodEntry(datetime.now(), description, mood, confidence)
        self.save_entry(entry)
        return entry

    def summarize(self) -> None:
        if not self.entries:
            print("No mood entries found yet. Add a mood entry first.")
            return
        counts = Counter(entry.mood for entry in self.entries)
        total_score = sum(MOOD_SCORE.get(entry.mood, 3) for entry in self.entries)
        average = total_score / len(self.entries)
        print("\nMood Summary")
        print("-------------")
        for mood in MOOD_LABELS:
            print(f"{mood:8}: {counts.get(mood, 0)}")
        print(f"Total entries: {len(self.entries)}")
        print(f"Average mood score: {average:.2f} / 5.00")
        top = counts.most_common(1)
        if top:
            print(f"Most frequent mood: {top[0][0]} ({top[0][1]} entries)")

    def show_entries(self, limit: Optional[int] = None) -> None:
        if not self.entries:
            print("No mood entries to display.")
            return
        entries = self.entries[-limit:] if limit else self.entries
        print("\nRecent mood entries")
        print("--------------------")
        for entry in entries:
            timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {entry.mood} ({entry.confidence:.0%}) - {entry.description}")


def prompt_input(prompt: str) -> str:
    value = input(prompt).strip()
    return value


def print_menu() -> None:
    print("\nAI/ML Mood Tracker")
    print("------------------")
    print("1. Add mood entry")
    print("2. Show mood summary")
    print("3. Show recent entries")
    print("4. Exit")


def main() -> None:
    tracker = MoodTracker()
    while True:
        print_menu()
        choice = prompt_input("Choose an option: ")
        if choice == "1":
            description = prompt_input("Describe how you feel right now: ")
            if not description:
                print("Please enter a short description of your mood.")
                continue
            entry = tracker.add_entry(description)
            print(
                f"Mood saved: {entry.mood} ({entry.confidence:.0%})\n"
                f"Description: {entry.description}"
            )
        elif choice == "2":
            tracker.summarize()
        elif choice == "3":
            tracker.show_entries(limit=10)
        elif choice == "4":
            print("Goodbye. Keep tracking your mood!")
            break
        else:
            print("Invalid option. Enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()