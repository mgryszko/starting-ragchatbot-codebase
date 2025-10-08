"""
Shared pytest fixtures and configuration for all tests
"""

import pytest
import sys
import os
import tempfile
import shutil

# Add parent directory to path so tests can import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore
from config import Config


@pytest.fixture(scope="session")
def sample_course_data():
    """
    Reusable sample course data for tests
    """
    course = Course(
        title="Introduction to AI",
        course_link="https://example.com/ai-course",
        instructor="Dr. Alan Turing",
        lessons=[
            Lesson(
                lesson_number=1,
                title="What is Artificial Intelligence?",
                lesson_link="https://example.com/ai-course/lesson1"
            ),
            Lesson(
                lesson_number=2,
                title="Machine Learning Basics",
                lesson_link="https://example.com/ai-course/lesson2"
            ),
            Lesson(
                lesson_number=3,
                title="Neural Networks",
                lesson_link="https://example.com/ai-course/lesson3"
            )
        ]
    )

    chunks = [
        CourseChunk(
            content="Artificial Intelligence is the simulation of human intelligence by machines.",
            course_title=course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Machine learning is a subset of AI that enables computers to learn from data.",
            course_title=course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Neural networks are computing systems inspired by biological neural networks.",
            course_title=course.title,
            lesson_number=3,
            chunk_index=2
        )
    ]

    return {"course": course, "chunks": chunks}


@pytest.fixture
def temp_directory():
    """
    Create a temporary directory for tests, cleaned up after use
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_directory):
    """
    Create a test configuration with temporary paths
    """
    config = Config()
    config.CHROMA_PATH = temp_directory
    config.ANTHROPIC_API_KEY = "test-api-key-placeholder"
    return config


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may be slow)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast)"
    )
