"""
Tests for CourseSearchTool.execute() method

These tests verify that the CourseSearchTool correctly executes searches
and returns properly formatted results.
"""

import pytest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search_tools import CourseSearchTool
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk
from unittest.mock import Mock, MagicMock, patch


class TestCourseSearchToolExecute:
    """Test suite for CourseSearchTool.execute() method"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store"""
        return Mock(spec=VectorStore)

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create a CourseSearchTool instance with mock vector store"""
        return CourseSearchTool(mock_vector_store)

    def test_execute_with_successful_search_results(self, search_tool, mock_vector_store):
        """Test that execute returns formatted results when search succeeds"""
        # Arrange
        mock_results = SearchResults(
            documents=["This is content about MCP", "More content about servers"],
            metadata=[
                {"course_title": "MCP Course", "lesson_number": 1},
                {"course_title": "MCP Course", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        # Act
        result = search_tool.execute(query="What is MCP?")

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="What is MCP?",
            course_name=None,
            lesson_number=None
        )
        assert isinstance(result, str)
        assert "MCP Course" in result
        assert "Lesson 1" in result
        assert "This is content about MCP" in result

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test that execute correctly passes course_name filter to vector store"""
        # Arrange
        mock_results = SearchResults(
            documents=["Course content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = search_tool.execute(query="MCP info", course_name="MCP")

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="MCP info",
            course_name="MCP",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test that execute correctly passes lesson_number filter to vector store"""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = search_tool.execute(query="lesson info", lesson_number=3)

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="lesson info",
            course_name=None,
            lesson_number=3
        )

    def test_execute_with_both_filters(self, search_tool, mock_vector_store):
        """Test that execute correctly passes both course and lesson filters"""
        # Arrange
        mock_results = SearchResults(
            documents=["Specific content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = search_tool.execute(query="specific info", course_name="MCP", lesson_number=2)

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="specific info",
            course_name="MCP",
            lesson_number=2
        )

    def test_execute_with_search_error(self, search_tool, mock_vector_store):
        """Test that execute returns error message when search fails"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search error: Database connection failed"
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="test query")

        # Assert
        assert result == "Search error: Database connection failed"

    def test_execute_with_no_results(self, search_tool, mock_vector_store):
        """Test that execute returns appropriate message when no results found"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="nonexistent content")

        # Assert
        assert "No relevant content found" in result

    def test_execute_no_results_with_course_filter(self, search_tool, mock_vector_store):
        """Test error message includes course name when filtered"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="test", course_name="Nonexistent Course")

        # Assert
        assert "No relevant content found" in result
        assert "Nonexistent Course" in result

    def test_execute_tracks_sources_for_ui(self, search_tool, mock_vector_store):
        """Test that execute stores source information in last_sources"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2"
        ]

        # Act
        result = search_tool.execute(query="test")

        # Assert
        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert search_tool.last_sources[0]["link"] == "https://example.com/lesson1"
        assert search_tool.last_sources[1]["text"] == "Course B - Lesson 2"
        assert search_tool.last_sources[1]["link"] == "https://example.com/lesson2"

    def test_execute_format_includes_all_metadata(self, search_tool, mock_vector_store):
        """Test that formatted output includes course and lesson information"""
        # Arrange
        mock_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 5}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = search_tool.execute(query="test")

        # Assert
        assert "[Test Course - Lesson 5]" in result
        assert "Test content" in result

    def test_execute_handles_missing_lesson_number(self, search_tool, mock_vector_store):
        """Test that execute handles metadata without lesson_number"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content without lesson"],
            metadata=[{"course_title": "General Course"}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        # Act
        result = search_tool.execute(query="test")

        # Assert
        assert "[General Course]" in result
        assert "Content without lesson" in result
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["text"] == "General Course"
