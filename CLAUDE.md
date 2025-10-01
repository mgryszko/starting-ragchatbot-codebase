# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

### Running the Application
```bash
# Set up environment first
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Run application (starts server on port 8000)
./run.sh

# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Installing Dependencies
```bash
uv sync
```

## System Architecture

### RAG Pipeline Overview
This is a Retrieval-Augmented Generation (RAG) system where Claude uses tool calling to dynamically search course materials. The pipeline follows this pattern:

1. **User Query** → Frontend (`frontend/script.js`)
2. **API Gateway** → FastAPI (`backend/app.py`)
3. **RAG Orchestrator** → `backend/rag_system.py`
4. **AI Generation** → `backend/ai_generator.py` (makes 1-2 Claude API calls)
5. **Tool Execution** → `backend/search_tools.py` (if Claude decides to search)
6. **Vector Search** → `backend/vector_store.py` (ChromaDB)
7. **Response** → Back through the chain to frontend

### Two-Pass AI Pattern
The system uses Claude's tool calling with a two-pass approach:
- **First API call**: Claude receives the query + tool definitions, decides whether to search
- **If tool used**: Execute search, gather results
- **Second API call**: Claude synthesizes final answer from search results

This happens in `ai_generator.py:43-135`, specifically the `_handle_tool_execution()` method.

### Data Storage: Dual Collection Strategy
The vector store (`vector_store.py`) uses **two ChromaDB collections**:

1. **`course_catalog`**: Course-level metadata (titles, instructors, lesson metadata)
   - Used for fuzzy course name matching via semantic search
   - Course title serves as the unique ID

2. **`course_content`**: Chunked lesson content
   - Searchable with filters: `course_title`, `lesson_number`
   - Each chunk tagged with course context

When a query mentions a course name (e.g., "MCP course"), the system:
1. Uses semantic search on `course_catalog` to resolve partial/fuzzy names → exact title
2. Filters `course_content` search by that exact title

### Document Processing Format
Course documents in `docs/` must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 1: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 2: [lesson title]
Lesson Link: [url]
[lesson content...]
```

The `DocumentProcessor` (`document_processor.py`) parses this format and creates:
- One `Course` object with embedded `Lesson` objects
- Multiple `CourseChunk` objects (800 char chunks with 100 char overlap)

### Session Management
Sessions maintain conversation context via `SessionManager` (`session_manager.py`):
- Stores last N exchanges (default: 2 exchanges = 4 messages)
- Session history is passed to Claude as system prompt context
- Enables multi-turn conversations with context retention

## Key Components

### `backend/rag_system.py` (Orchestrator)
Main entry point for query processing. Coordinates:
- Session management
- Tool registration
- AI generation
- Source tracking

Key method: `query(query, session_id)` at line 102

### `backend/ai_generator.py` (Claude Interface)
Handles all Claude API interactions with tool calling support.
- System prompt defines behavior at line 8
- Tool execution logic at line 89
- Temperature: 0, Max tokens: 800

### `backend/search_tools.py` (Tool System)
Implements Claude's tool calling interface:
- `Tool` abstract base class defines tool contract
- `CourseSearchTool` implements search with course/lesson filtering
- `ToolManager` handles tool registration and execution

Tool definition follows Anthropic's tool schema (line 27-50).

### `backend/vector_store.py` (Data Layer)
ChromaDB wrapper with semantic search capabilities:
- `search()` method (line 61) is the main interface
- Uses sentence-transformers model: `all-MiniLM-L6-v2`
- Course name resolution via `_resolve_course_name()` at line 102

### Configuration
All settings centralized in `backend/config.py`:
- Loads from `.env` file (requires `ANTHROPIC_API_KEY`)
- Model: `claude-sonnet-4-20250514`
- Chunk size: 800 chars, overlap: 100 chars
- Max results: 5, Max history: 2 exchanges

## Frontend Architecture
Single-page application (`frontend/`):
- `script.js`: Handles API communication, markdown rendering (uses marked.js)
- Posts to `/api/query` endpoint (line 63)
- Displays sources in collapsible sections
- Session ID maintained in `currentSessionId` variable

## Important Patterns

### Adding New Tools
To add a tool that Claude can use:
1. Create class implementing `Tool` interface in `search_tools.py`
2. Implement `get_tool_definition()` with Anthropic tool schema
3. Implement `execute(**kwargs)` method
4. Register in `rag_system.py:24` via `tool_manager.register_tool()`

### Modifying Search Behavior
Search logic is in `vector_store.py:61-100`:
- Course name filtering uses semantic matching (not exact match)
- Lesson filtering is exact integer match
- Both filters can be combined with `$and` operator

### Document Loading
Documents are loaded on app startup (`app.py:88-98`):
- Checks for existing courses to avoid duplicates
- Uses course title as deduplication key
- Can be forced to rebuild with `clear_existing=True`
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies