# RAG System Test Suite

## Overview

This directory contains comprehensive tests for the RAG chatbot system, covering all major components from individual units to full end-to-end integration.

## Test Files

### 1. `test_vector_store.py` (25 tests)
Tests the ChromaDB vector store integration with real database operations.

**Coverage:**
- Course metadata storage and retrieval
- Content chunk storage
- Semantic search with and without filters
- Course name resolution (fuzzy matching)
- Edge cases (empty database, missing data, etc.)

**Key Tests:**
- Partial course name matching
- Lesson number filtering
- Multi-course handling
- Data clearing operations

### 2. `test_course_search_tool.py` (10 tests)
Tests the CourseSearchTool execution and result formatting.

**Coverage:**
- Tool definition format (Anthropic schema)
- Query parameter passing
- Result formatting with metadata
- Source tracking for UI
- Error handling

**Key Tests:**
- Successful search results
- Filter combinations (course + lesson)
- Empty results handling
- Source link generation

### 3. `test_ai_generator.py` (10 tests)
Tests Claude API integration and tool calling flow.

**Coverage:**
- Basic response generation
- Tool definition passing to Claude
- Tool execution workflow (two-pass pattern)
- Message formatting for multi-turn conversations
- Error propagation

**Key Tests:**
- Tool use detection and execution
- Multiple tool calls in single response
- Conversation history integration
- Temperature and token settings

### 4. `test_rag_integration.py` (18 tests)
End-to-end integration tests of the complete RAG pipeline.

**Coverage:**
- Full query flow from request to response
- Session management
- Tool manager coordination
- Source tracking and reset
- Multi-course scenarios

**Key Tests:**
- Complete query pipeline with mocked API
- Session isolation
- Document loading from files
- Analytics and course statistics

### 5. `conftest.py`
Shared pytest fixtures and configuration.

**Fixtures:**
- `sample_course_data` - Reusable test data
- `temp_directory` - Temporary file storage
- `test_config` - Test configuration with temp paths

## Running Tests

### All Tests
```bash
cd backend
uv run pytest tests/ -v
```

### Specific Test File
```bash
uv run pytest tests/test_vector_store.py -v
```

### Specific Test
```bash
uv run pytest tests/test_vector_store.py::TestVectorStoreIntegration::test_search_without_filters -v
```

### With Coverage Report
```bash
uv run pytest tests/ --cov=. --cov-report=html
```

## Test Results

**Latest Run:** 55 tests total
- ✅ **52 passed** (94.5%)
- ❌ **3 failed** (5.5%)

### Failures Identified

1. **test_search_nonexistent_course** - Revealed critical bug in course name resolution
2. **test_query_with_invalid_course_name** - Same root cause as above
3. **test_max_results_limit** - Test implementation issue (ChromaDB API change)

See `TEST_RESULTS.md` for detailed analysis.

## Critical Findings

### 🔴 Bug: Course Name Resolution Too Permissive

**Problem:** Vector store accepts ANY course name match, even with terrible similarity scores.

**Impact:** Users querying non-existent courses get results from wrong courses, appearing as "query failed"

**Location:** `vector_store.py:102-116` (`_resolve_course_name()`)

**Fix:** Add similarity threshold check to reject poor matches

See `PROPOSED_FIXES.md` for complete solution.

## Test Architecture

### Component Testing Strategy

```
┌─────────────────────────────────────────────────────┐
│                  User Query                          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  RAGSystem Integration Tests (test_rag_integration) │
│  - Full pipeline                                     │
│  - Session management                                │
│  - Tool coordination                                 │
└─────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌──────────────────┐          ┌──────────────────────┐
│  AIGenerator     │          │  CourseSearchTool    │
│  Tests           │          │  Tests               │
│  (test_ai_gen)   │          │  (test_course_search)│
└──────────────────┘          └──────────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  VectorStore Tests            │
        │  (test_vector_store)          │
        │  - Real ChromaDB integration  │
        └───────────────────────────────┘
```

### Test Isolation

- **Unit tests** use mocks to isolate components
- **Integration tests** use temporary ChromaDB instances
- **No tests hit production database**
- **Fixtures ensure cleanup** after each test

## Adding New Tests

### Template for New Test File

```python
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from your_module import YourClass


class TestYourFeature:
    """Test suite for YourFeature"""

    @pytest.fixture
    def your_fixture(self):
        """Create test instance"""
        return YourClass()

    def test_basic_functionality(self, your_fixture):
        """Test description"""
        # Arrange
        expected = "result"

        # Act
        result = your_fixture.do_something()

        # Assert
        assert result == expected
```

### Best Practices

1. **Use fixtures** for common setup
2. **One assertion per test** when possible
3. **Descriptive test names** that explain what's being tested
4. **Docstrings** explaining the test purpose
5. **Arrange-Act-Assert** pattern
6. **Mock external services** (API calls, file I/O)
7. **Clean up resources** in fixtures

## Dependencies

```toml
pytest>=8.0.0
pytest-mock>=3.12.0
pytest-asyncio>=0.23.0
```

Installed via: `uv sync`

## Continuous Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    cd backend
    uv sync
    uv run pytest tests/ -v --junitxml=test-results.xml
```

## Troubleshooting

### Tests Fail with "Module not found"
```bash
# Ensure you're in the backend directory
cd backend
uv run pytest tests/
```

### ChromaDB Permission Errors
```bash
# Clean up temporary test databases
rm -rf /tmp/pytest-*
```

### Import Errors
```bash
# Reinstall dependencies
uv sync --reinstall
```

## Next Steps

1. ✅ Tests written and documented
2. ✅ Bugs identified
3. ⏭️ Implement fixes from `PROPOSED_FIXES.md`
4. ⏭️ Re-run tests to verify fixes
5. ⏭️ Deploy to production

## Contact

For questions about the test suite, see:
- `TEST_RESULTS.md` - Detailed test analysis
- `PROPOSED_FIXES.md` - Fix proposals and implementation guide
