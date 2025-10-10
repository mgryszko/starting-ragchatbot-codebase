"""
Shared pytest fixtures and configuration for all tests
"""

import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock

import pytest

# Add parent directory to path so tests can import backend modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from models import Course, CourseChunk, Lesson
from vector_store import VectorStore


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
                lesson_link="https://example.com/ai-course/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Machine Learning Basics",
                lesson_link="https://example.com/ai-course/lesson2",
            ),
            Lesson(
                lesson_number=3,
                title="Neural Networks",
                lesson_link="https://example.com/ai-course/lesson3",
            ),
        ],
    )

    chunks = [
        CourseChunk(
            content="Artificial Intelligence is the simulation of human intelligence by machines.",
            course_title=course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Machine learning is a subset of AI that enables computers to learn from data.",
            course_title=course.title,
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Neural networks are computing systems inspired by biological neural networks.",
            course_title=course.title,
            lesson_number=3,
            chunk_index=2,
        ),
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
    config.addinivalue_line("markers", "unit: mark test as unit test (fast)")
    config.addinivalue_line("markers", "api: mark test as API endpoint test")


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def mock_rag_system():
    """
    Mock RAG system for API testing
    Returns a mock with predefined responses
    """
    mock = MagicMock()

    # Default query response
    mock.query.return_value = (
        "This is a test response from the RAG system.",
        [
            {
                "course_title": "Introduction to AI",
                "lesson_number": 1,
                "lesson_link": "https://example.com/ai-course/lesson1",
                "content": "Sample content chunk"
            }
        ]
    )

    # Default analytics response
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to AI", "Machine Learning Basics"]
    }

    # Mock session manager
    mock.session_manager = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-123"
    mock.session_manager.clear_session.return_value = None

    return mock


@pytest.fixture
def test_app(mock_rag_system):
    """
    Create a test FastAPI app without static file mounting
    This avoids issues with missing frontend directory in tests
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any

    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class ClearSessionRequest(BaseModel):
        session_id: str

    # Define endpoints inline
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/clear")
    async def clear_session(request: ClearSessionRequest):
        try:
            mock_rag_system.session_manager.clear_session(request.session_id)
            return {"status": "success", "message": "Session cleared"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def client(test_app):
    """
    Create a test client for API testing
    """
    from fastapi.testclient import TestClient
    return TestClient(test_app)
