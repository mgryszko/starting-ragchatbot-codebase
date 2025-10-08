"""
Integration tests for the complete RAG system

These tests verify the end-to-end flow from query to response.
"""

import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk
from ai_generator import AIGenerator
from vector_store import VectorStore


class TestRAGSystemIntegration:
    """End-to-end integration tests for RAG system"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary ChromaDB directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_config(self, temp_chroma_path):
        """Create test configuration"""
        config = Config()
        config.CHROMA_PATH = temp_chroma_path
        config.ANTHROPIC_API_KEY = "test-api-key-12345"
        return config

    @pytest.fixture
    def rag_system(self, test_config):
        """Create RAG system instance"""
        return RAGSystem(test_config)

    @pytest.fixture
    def sample_course(self):
        """Create sample course for testing"""
        return Course(
            title="Model Context Protocol Course",
            course_link="https://example.com/mcp",
            instructor="Dr. Test",
            lessons=[
                Lesson(lesson_number=1, title="Introduction to MCP", lesson_link="https://example.com/mcp/1"),
                Lesson(lesson_number=2, title="MCP Servers", lesson_link="https://example.com/mcp/2")
            ]
        )

    @pytest.fixture
    def sample_chunks(self, sample_course):
        """Create sample chunks"""
        return [
            CourseChunk(
                content="The Model Context Protocol (MCP) is a standard for connecting AI assistants to data sources.",
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="MCP servers provide tools and resources that AI assistants can use to access external systems.",
                course_title=sample_course.title,
                lesson_number=2,
                chunk_index=1
            )
        ]

    def test_rag_system_initialization(self, rag_system):
        """Test that RAG system initializes all components"""
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None

    def test_add_course_to_system(self, rag_system, sample_course, sample_chunks):
        """Test adding course metadata and content"""
        # Add course
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_chunks)

        # Verify
        assert rag_system.vector_store.get_course_count() == 1
        titles = rag_system.vector_store.get_existing_course_titles()
        assert sample_course.title in titles

    def test_tool_manager_has_search_tool(self, rag_system):
        """Test that tool manager has search tool registered"""
        tool_defs = rag_system.tool_manager.get_tool_definitions()

        # Should have at least the search tool
        assert len(tool_defs) > 0
        tool_names = [tool['name'] for tool in tool_defs]
        assert 'search_course_content' in tool_names

    def test_tool_manager_has_outline_tool(self, rag_system):
        """Test that tool manager has outline tool registered"""
        tool_defs = rag_system.tool_manager.get_tool_definitions()

        tool_names = [tool['name'] for tool in tool_defs]
        assert 'get_course_outline' in tool_names

    def test_search_tool_execution_via_tool_manager(self, rag_system, sample_course, sample_chunks):
        """Test executing search tool through tool manager"""
        # Setup data
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_chunks)

        # Execute search via tool manager
        result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="What is MCP?"
        )

        # Verify result
        assert isinstance(result, str)
        assert "MCP" in result or "Model Context Protocol" in result

    def test_query_without_api_call(self, rag_system, sample_course, sample_chunks):
        """Test query flow without actually calling Anthropic API"""
        # Setup data
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_chunks)

        # Mock AI generator to avoid API call
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            mock_generate.return_value = "MCP is a protocol for AI assistants."

            # Execute query
            response, sources = rag_system.query("What is MCP?")

            # Verify
            assert response == "MCP is a protocol for AI assistants."
            mock_generate.assert_called_once()

            # Check that tools were passed
            call_args = mock_generate.call_args
            assert 'tools' in call_args.kwargs
            assert 'tool_manager' in call_args.kwargs

    def test_query_with_session_id(self, rag_system):
        """Test query with session management"""
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            mock_generate.return_value = "Response"

            # First query with session
            session_id = rag_system.session_manager.create_session()
            response1, _ = rag_system.query("First question", session_id=session_id)

            # Second query with same session
            response2, _ = rag_system.query("Follow-up question", session_id=session_id)

            # Verify history was used in second call
            second_call = mock_generate.call_args_list[1]
            assert 'conversation_history' in second_call.kwargs
            history = second_call.kwargs['conversation_history']
            assert history is not None
            assert "First question" in history

    def test_source_tracking_from_search(self, rag_system, sample_course, sample_chunks):
        """Test that sources are tracked from search results"""
        # Setup data
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_chunks)

        # Execute search directly to populate sources
        result = rag_system.search_tool.execute(query="MCP")

        # Check that sources were tracked
        assert len(rag_system.search_tool.last_sources) > 0

        # Verify source format
        source = rag_system.search_tool.last_sources[0]
        assert 'text' in source
        assert 'link' in source

    def test_source_reset_after_retrieval(self, rag_system, sample_course, sample_chunks):
        """Test that sources are reset after being retrieved"""
        # Setup data
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_chunks)

        # Execute search
        rag_system.search_tool.execute(query="MCP")

        # Get sources via tool manager
        sources = rag_system.tool_manager.get_last_sources()
        assert len(sources) > 0

        # Reset sources
        rag_system.tool_manager.reset_sources()

        # Verify sources are cleared
        sources_after = rag_system.tool_manager.get_last_sources()
        assert len(sources_after) == 0

    def test_course_analytics(self, rag_system, sample_course, sample_chunks):
        """Test getting course analytics"""
        # Setup data
        rag_system.vector_store.add_course_metadata(sample_course)

        # Get analytics
        analytics = rag_system.get_course_analytics()

        # Verify
        assert 'total_courses' in analytics
        assert 'course_titles' in analytics
        assert analytics['total_courses'] == 1
        assert sample_course.title in analytics['course_titles']

    def test_add_course_document_from_file(self, rag_system, temp_chroma_path):
        """Test adding course from document file"""
        # Create a temporary course document
        doc_path = os.path.join(temp_chroma_path, "test_course.txt")
        with open(doc_path, 'w') as f:
            f.write("""Course Title: Test Course
Course Link: https://example.com/test
Course Instructor: Dr. Test

Lesson 1: Introduction
Lesson Link: https://example.com/lesson1
This is lesson 1 content about testing.

Lesson 2: Advanced Topics
Lesson Link: https://example.com/lesson2
This is lesson 2 content about advanced testing.
""")

        # Add document
        course, chunk_count = rag_system.add_course_document(doc_path)

        # Verify
        assert course is not None
        assert course.title == "Test Course"
        assert chunk_count > 0
        assert rag_system.vector_store.get_course_count() == 1

    def test_empty_database_query(self, rag_system):
        """Test querying with empty database"""
        # Don't add any data

        # Mock AI to test behavior
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            # Simulate tool being called but finding nothing
            def mock_response(query, conversation_history, tools, tool_manager):
                # Simulate AI deciding to search
                result = tool_manager.execute_tool("search_course_content", query="test")
                # AI should get empty results
                return "I couldn't find any information about that."

            mock_generate.side_effect = mock_response

            response, sources = rag_system.query("What is MCP?")

            # Should handle gracefully
            assert isinstance(response, str)
            assert len(sources) == 0

    def test_query_with_invalid_course_name(self, rag_system, sample_course, sample_chunks):
        """Test searching for non-existent course"""
        # Add real course
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_chunks)

        # Search for non-existent course via tool manager
        # Use a clearly nonsensical name that won't match anything
        result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="anything",
            course_name="XYZABC12345 Fake Nonsense"
        )

        # Should return error message
        assert "No course found" in result

    def test_outline_tool_execution(self, rag_system, sample_course):
        """Test getting course outline via tool manager"""
        # Add course
        rag_system.vector_store.add_course_metadata(sample_course)

        # Get outline via tool manager
        result = rag_system.tool_manager.execute_tool(
            "get_course_outline",
            course_name=sample_course.title
        )

        # Verify
        assert isinstance(result, str)
        assert sample_course.title in result
        assert sample_course.instructor in result
        assert "Introduction to MCP" in result

    def test_multiple_courses_search(self, rag_system, sample_course):
        """Test searching across multiple courses"""
        # Add first course
        rag_system.vector_store.add_course_metadata(sample_course)
        chunks1 = [
            CourseChunk(
                content="MCP enables AI assistants to connect to data sources",
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=0
            )
        ]
        rag_system.vector_store.add_course_content(chunks1)

        # Add second course
        course2 = Course(
            title="Deep Learning Basics",
            course_link="https://example.com/dl",
            instructor="Dr. Neural",
            lessons=[Lesson(lesson_number=1, title="Neural Networks")]
        )
        rag_system.vector_store.add_course_metadata(course2)
        chunks2 = [
            CourseChunk(
                content="Neural networks are the foundation of deep learning",
                course_title=course2.title,
                lesson_number=1,
                chunk_index=0
            )
        ]
        rag_system.vector_store.add_course_content(chunks2)

        # Search without filter (should search both)
        result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="learning"
        )

        # Should find results
        assert isinstance(result, str)
        assert len(result) > 0

    def test_session_isolation(self, rag_system):
        """Test that different sessions maintain separate histories"""
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            mock_generate.return_value = "Response"

            # Create two sessions
            session1 = rag_system.session_manager.create_session()
            session2 = rag_system.session_manager.create_session()

            # Query in session 1
            rag_system.query("Question 1", session_id=session1)
            rag_system.query("Question 2", session_id=session1)

            # Query in session 2
            rag_system.query("Question A", session_id=session2)

            # Check session 1 history
            history1 = rag_system.session_manager.get_conversation_history(session1)
            assert "Question 1" in history1
            assert "Question A" not in history1

            # Check session 2 history
            history2 = rag_system.session_manager.get_conversation_history(session2)
            assert "Question A" in history2
            assert "Question 1" not in history2

    def test_tool_execution_error_handling(self, rag_system):
        """Test handling of tool execution errors"""
        # Try to execute non-existent tool
        result = rag_system.tool_manager.execute_tool(
            "nonexistent_tool",
            query="test"
        )

        # Should return error message
        assert "not found" in result

    def test_query_formats_prompt_correctly(self, rag_system):
        """Test that query properly formats the prompt"""
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_generate:
            mock_generate.return_value = "Response"

            rag_system.query("What is MCP?")

            # Check the prompt passed to AI
            call_args = mock_generate.call_args
            query_arg = call_args.kwargs['query']
            assert "What is MCP?" in query_arg
            assert "course materials" in query_arg.lower()
