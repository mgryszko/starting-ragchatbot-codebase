"""
API Endpoint Tests

Tests for FastAPI endpoints in the RAG system:
- /api/query (POST): Process queries and return responses with sources
- /api/courses (GET): Retrieve course statistics and metadata
- /api/session/clear (POST): Clear conversation session history

These tests use a mock RAG system to avoid dependencies on external services
and vector database operations.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.mark.api
class TestQueryEndpoint:
    """Tests for the /api/query endpoint"""

    def test_query_without_session_id(self, client, mock_rag_system):
        """Test querying without providing a session ID creates a new session"""
        # Arrange
        request_data = {
            "query": "What is artificial intelligence?"
        }

        # Act
        response = client.post("/api/query", json=request_data)

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"  # From mock
        assert data["answer"] == "This is a test response from the RAG system."

        # Verify session was created
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_existing_session_id(self, client, mock_rag_system):
        """Test querying with an existing session ID uses that session"""
        # Arrange
        request_data = {
            "query": "Tell me more about neural networks",
            "session_id": "existing-session-456"
        }

        # Act
        response = client.post("/api/query", json=request_data)

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == "existing-session-456"

        # Verify no new session was created
        mock_rag_system.session_manager.create_session.assert_not_called()

        # Verify query was called with the session ID
        mock_rag_system.query.assert_called_once_with(
            "Tell me more about neural networks",
            "existing-session-456"
        )

    def test_query_returns_sources(self, client, mock_rag_system):
        """Test that query response includes source information"""
        # Arrange
        request_data = {"query": "What is machine learning?"}

        # Act
        response = client.post("/api/query", json=request_data)

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

        source = data["sources"][0]
        assert "course_title" in source
        assert "lesson_number" in source
        assert "lesson_link" in source
        assert "content" in source

    def test_query_with_empty_query_string(self, client):
        """Test that empty query strings are handled"""
        # Arrange
        request_data = {"query": ""}

        # Act
        response = client.post("/api/query", json=request_data)

        # Assert
        # Should still process (validation is at RAG system level)
        assert response.status_code == 200

    def test_query_missing_query_field(self, client):
        """Test that missing query field returns validation error"""
        # Arrange
        request_data = {"session_id": "test-123"}  # Missing query field

        # Act
        response = client.post("/api/query", json=request_data)

        # Assert
        assert response.status_code == 422  # Unprocessable Entity

    def test_query_with_invalid_json(self, client):
        """Test that invalid JSON returns error"""
        # Act
        response = client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        # Assert
        assert response.status_code == 422

    def test_query_exception_handling(self, client, mock_rag_system):
        """Test that exceptions in RAG system are handled properly"""
        # Arrange
        mock_rag_system.query.side_effect = Exception("Database connection failed")
        request_data = {"query": "What is AI?"}

        # Act
        response = client.post("/api/query", json=request_data)

        # Assert
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Tests for the /api/courses endpoint"""

    def test_get_courses_success(self, client, mock_rag_system):
        """Test successful retrieval of course statistics"""
        # Act
        response = client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert isinstance(data["course_titles"], list)
        assert len(data["course_titles"]) == 2

    def test_get_courses_includes_course_titles(self, client, mock_rag_system):
        """Test that course titles are returned correctly"""
        # Act
        response = client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert "Introduction to AI" in data["course_titles"]
        assert "Machine Learning Basics" in data["course_titles"]

    def test_get_courses_empty_catalog(self, client, mock_rag_system):
        """Test response when no courses exist"""
        # Arrange
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        # Act
        response = client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_exception_handling(self, client, mock_rag_system):
        """Test that exceptions are handled properly"""
        # Arrange
        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store error")

        # Act
        response = client.get("/api/courses")

        # Assert
        assert response.status_code == 500
        assert "Vector store error" in response.json()["detail"]


@pytest.mark.api
class TestSessionClearEndpoint:
    """Tests for the /api/session/clear endpoint"""

    def test_clear_session_success(self, client, mock_rag_system):
        """Test successful session clearing"""
        # Arrange
        request_data = {"session_id": "session-to-clear"}

        # Act
        response = client.post("/api/session/clear", json=request_data)

        # Assert
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert data["message"] == "Session cleared"

        # Verify session manager was called
        mock_rag_system.session_manager.clear_session.assert_called_once_with(
            "session-to-clear"
        )

    def test_clear_session_missing_session_id(self, client):
        """Test that missing session_id returns validation error"""
        # Arrange
        request_data = {}  # Missing session_id

        # Act
        response = client.post("/api/session/clear", json=request_data)

        # Assert
        assert response.status_code == 422  # Unprocessable Entity

    def test_clear_session_exception_handling(self, client, mock_rag_system):
        """Test that exceptions are handled properly"""
        # Arrange
        mock_rag_system.session_manager.clear_session.side_effect = Exception(
            "Session not found"
        )
        request_data = {"session_id": "nonexistent-session"}

        # Act
        response = client.post("/api/session/clear", json=request_data)

        # Assert
        assert response.status_code == 500
        assert "Session not found" in response.json()["detail"]


@pytest.mark.api
class TestCORSHeaders:
    """Tests for CORS configuration"""

    def test_cors_middleware_configured(self, test_app):
        """
        Test that middleware is configured in the app
        Note: TestClient doesn't trigger CORS headers like a real browser would,
        so we just verify that the app has middleware configured
        """
        # Check that middleware stack is not empty
        assert len(test_app.user_middleware) > 0

    def test_cors_preflight_request(self, client):
        """Test CORS preflight (OPTIONS) requests are handled"""
        # Act
        response = client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )

        # Assert
        # OPTIONS requests should be handled successfully
        assert response.status_code == 200


@pytest.mark.api
class TestResponseModels:
    """Tests for response model validation"""

    def test_query_response_schema(self, client):
        """Test that query response matches expected schema"""
        # Arrange
        request_data = {"query": "test query"}

        # Act
        response = client.post("/api/query", json=request_data)

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate schema
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_courses_response_schema(self, client):
        """Test that courses response matches expected schema"""
        # Act
        response = client.get("/api/courses")

        # Assert
        assert response.status_code == 200
        data = response.json()

        # Validate schema
        assert "total_courses" in data
        assert "course_titles" in data

        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # All titles should be strings
        for title in data["course_titles"]:
            assert isinstance(title, str)
