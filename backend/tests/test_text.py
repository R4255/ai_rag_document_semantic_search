import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_service import _tidy_text  # noqa: E402

# The shape pypdf produced for the Take-Home Assignment PDF: one word per line,
# a single blank line between words and two or more between paragraphs.
WORD_PER_LINE = (
    "Take-Home\n \nAssignment:\n \nAI-Powered\n \nMicro-CRM\n \nBuild\n \na\n \nsimple\n \ntool.\n \n \n"
    "Use\n \nyour\n \njudgment:\n \n \n-\n \nWhich\n \nproblems\n \n-\n \nWhere\n \nAI\n \nhelps"
)


def test_word_per_line_text_is_rejoined():
    assert _tidy_text(WORD_PER_LINE) == (
        "Take-Home Assignment: AI-Powered Micro-CRM Build a simple tool.\n\n"
        "Use your judgment:\n\n"
        "- Which problems\n- Where AI helps"
    )


def test_normal_text_is_left_alone():
    text = "Kafka partitions limit consumer parallelism.\nEach partition has one consumer.\n\nRedis caches hot reads."
    assert _tidy_text(text) == text


def test_short_text_is_left_alone():
    assert _tidy_text("Title\n\nOne\nTwo") == "Title\n\nOne\nTwo"
