"""
Tests for VectorStore with real ChromaDB integration

These tests verify the vector store correctly stores and retrieves course data.
"""

import pytest
import sys
import os
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStoreIntegration:
    """Integration tests for VectorStore with real ChromaDB"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create a temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def vector_store(self, temp_chroma_path):
        """Create a VectorStore instance with temporary database"""
        return VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

    @pytest.fixture
    def sample_course(self):
        """Create a sample course with lessons"""
        return Course(
            title="Introduction to Machine Learning",
            course_link="https://example.com/ml-course",
            instructor="Dr. Smith",
            lessons=[
                Lesson(lesson_number=1, title="What is ML?", lesson_link="https://example.com/lesson1"),
                Lesson(lesson_number=2, title="Supervised Learning", lesson_link="https://example.com/lesson2"),
                Lesson(lesson_number=3, title="Unsupervised Learning", lesson_link="https://example.com/lesson3")
            ]
        )

    @pytest.fixture
    def sample_chunks(self, sample_course):
        """Create sample course chunks"""
        return [
            CourseChunk(
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Supervised learning uses labeled data to train models for prediction tasks.",
                course_title=sample_course.title,
                lesson_number=2,
                chunk_index=1
            ),
            CourseChunk(
                content="Unsupervised learning finds patterns in unlabeled data through clustering.",
                course_title=sample_course.title,
                lesson_number=3,
                chunk_index=2
            )
        ]

    def test_add_and_retrieve_course_metadata(self, vector_store, sample_course):
        """Test that course metadata is stored and can be retrieved"""
        # Add course metadata
        vector_store.add_course_metadata(sample_course)

        # Verify course count
        assert vector_store.get_course_count() == 1

        # Verify course titles
        titles = vector_store.get_existing_course_titles()
        assert sample_course.title in titles

    def test_add_course_content_chunks(self, vector_store, sample_course, sample_chunks):
        """Test that course content chunks are stored"""
        # Add metadata and content
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search for content
        results = vector_store.search("What is machine learning?")

        # Verify results
        assert not results.is_empty()
        assert len(results.documents) > 0
        assert results.error is None

    def test_search_without_filters(self, vector_store, sample_course, sample_chunks):
        """Test basic search without any filters"""
        # Setup
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search
        results = vector_store.search("supervised learning")

        # Verify
        assert not results.is_empty()
        assert any("supervised" in doc.lower() for doc in results.documents)

    def test_search_with_course_name_filter(self, vector_store, sample_course, sample_chunks):
        """Test search filtered by course name"""
        # Setup
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search with exact course name
        results = vector_store.search(
            query="learning",
            course_name=sample_course.title
        )

        # Verify
        assert not results.is_empty()
        for meta in results.metadata:
            assert meta['course_title'] == sample_course.title

    def test_search_with_partial_course_name(self, vector_store, sample_course, sample_chunks):
        """Test that partial course names work via semantic search"""
        # Setup
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search with partial name
        results = vector_store.search(
            query="learning",
            course_name="Machine Learning"  # Partial match
        )

        # Should still find results
        assert not results.is_empty()

    def test_search_with_lesson_number_filter(self, vector_store, sample_course, sample_chunks):
        """Test search filtered by lesson number"""
        # Setup
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search for lesson 2 specifically
        results = vector_store.search(
            query="learning",
            lesson_number=2
        )

        # Verify
        assert not results.is_empty()
        for meta in results.metadata:
            assert meta['lesson_number'] == 2

    def test_search_with_both_filters(self, vector_store, sample_course, sample_chunks):
        """Test search with both course and lesson filters"""
        # Setup
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search with both filters
        results = vector_store.search(
            query="learning",
            course_name=sample_course.title,
            lesson_number=3
        )

        # Verify
        assert not results.is_empty()
        for meta in results.metadata:
            assert meta['course_title'] == sample_course.title
            assert meta['lesson_number'] == 3

    def test_search_nonexistent_course(self, vector_store, sample_course, sample_chunks):
        """Test search for course that doesn't exist"""
        # Setup
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search for non-existent course
        results = vector_store.search(
            query="learning",
            course_name="Nonexistent Course XYZ"
        )

        # Should return error
        assert results.error is not None
        assert "No course found" in results.error

    def test_search_empty_database(self, vector_store):
        """Test search on empty database returns no results"""
        results = vector_store.search("anything")

        # Should return empty results, not error
        assert results.is_empty()

    def test_course_name_resolution(self, vector_store, sample_course):
        """Test that _resolve_course_name works correctly"""
        # Add course
        vector_store.add_course_metadata(sample_course)

        # Test exact match
        resolved = vector_store._resolve_course_name(sample_course.title)
        assert resolved == sample_course.title

        # Test partial match
        resolved = vector_store._resolve_course_name("Machine Learning")
        assert resolved == sample_course.title

    def test_get_course_outline(self, vector_store, sample_course):
        """Test retrieving complete course outline"""
        # Add course
        vector_store.add_course_metadata(sample_course)

        # Get outline
        outline = vector_store.get_course_outline(sample_course.title)

        # Verify
        assert outline is not None
        assert outline['course_title'] == sample_course.title
        assert outline['course_link'] == sample_course.course_link
        assert outline['instructor'] == sample_course.instructor
        assert len(outline['lessons']) == 3

    def test_get_course_outline_partial_name(self, vector_store, sample_course):
        """Test getting outline with partial course name"""
        # Add course
        vector_store.add_course_metadata(sample_course)

        # Get outline with partial name
        outline = vector_store.get_course_outline("Machine Learning")

        # Should still find it
        assert outline is not None
        assert outline['course_title'] == sample_course.title

    def test_get_lesson_link(self, vector_store, sample_course):
        """Test retrieving specific lesson link"""
        # Add course
        vector_store.add_course_metadata(sample_course)

        # Get lesson link
        link = vector_store.get_lesson_link(sample_course.title, 2)

        # Verify
        assert link == "https://example.com/lesson2"

    def test_get_lesson_link_nonexistent(self, vector_store, sample_course):
        """Test retrieving link for nonexistent lesson"""
        # Add course
        vector_store.add_course_metadata(sample_course)

        # Try to get non-existent lesson
        link = vector_store.get_lesson_link(sample_course.title, 99)

        # Should return None
        assert link is None

    def test_multiple_courses(self, vector_store, sample_course):
        """Test handling multiple courses"""
        # Add first course
        vector_store.add_course_metadata(sample_course)

        # Add second course
        course2 = Course(
            title="Deep Learning Fundamentals",
            course_link="https://example.com/dl-course",
            instructor="Dr. Jones",
            lessons=[
                Lesson(lesson_number=1, title="Neural Networks", lesson_link="https://example.com/dl-lesson1")
            ]
        )
        vector_store.add_course_metadata(course2)

        # Verify both exist
        assert vector_store.get_course_count() == 2
        titles = vector_store.get_existing_course_titles()
        assert sample_course.title in titles
        assert course2.title in titles

    def test_clear_all_data(self, vector_store, sample_course, sample_chunks):
        """Test clearing all data from vector store"""
        # Add data
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Verify data exists
        assert vector_store.get_course_count() > 0

        # Clear
        vector_store.clear_all_data()

        # Verify empty
        assert vector_store.get_course_count() == 0

        # Search should return empty
        results = vector_store.search("anything")
        assert results.is_empty()

    def test_max_results_limit(self, temp_chroma_path, sample_course):
        """Test that max_results parameter limits search results"""
        # Create store with limit of 2
        limited_store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=2
        )

        # Add course with many chunks
        limited_store.add_course_metadata(sample_course)
        chunks = [
            CourseChunk(
                content=f"Content piece {i} about machine learning and AI",
                course_title=sample_course.title,
                lesson_number=1,
                chunk_index=i
            )
            for i in range(10)
        ]
        limited_store.add_course_content(chunks)

        # Search
        results = limited_store.search("machine learning")

        # Should return max 2 results
        assert len(results.documents) <= 2
