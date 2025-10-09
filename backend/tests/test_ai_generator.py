"""
Tests for AIGenerator and Claude API tool calling integration

These tests verify the AI generator correctly handles tool calling with Claude.
"""

import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager


class MockContentBlock:
    """Mock for Anthropic content block"""

    def __init__(self, content_type, text=None, tool_use_data=None):
        self.type = content_type
        self.text = text
        if tool_use_data:
            self.name = tool_use_data["name"]
            self.id = tool_use_data["id"]
            self.input = tool_use_data["input"]


class MockResponse:
    """Mock for Anthropic API response"""

    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class TestAIGeneratorToolCalling:
    """Test suite for AIGenerator with tool calling"""

    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator instance with dummy API key"""
        return AIGenerator(api_key="test-api-key", model="claude-sonnet-4-20250514")

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock tool manager"""
        manager = Mock(spec=ToolManager)
        manager.execute_tool = Mock(
            return_value="Search result: Machine learning is a field of AI"
        )
        return manager

    @pytest.fixture
    def sample_tool_definitions(self):
        """Sample tool definitions"""
        return [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"}
                    },
                    "required": ["query"],
                },
            }
        ]

    def test_generate_response_without_tools(self, ai_generator):
        """Test basic response generation without tools"""
        with patch.object(ai_generator.client.messages, "create") as mock_create:
            # Mock response without tool use
            mock_create.return_value = MockResponse(
                stop_reason="end_turn",
                content=[MockContentBlock("text", text="This is a test response")],
            )

            # Generate response
            response = ai_generator.generate_response(
                query="What is AI?",
                conversation_history=None,
                tools=None,
                tool_manager=None,
            )

            # Verify
            assert response == "This is a test response"
            mock_create.assert_called_once()

            # Verify no tools were passed
            call_args = mock_create.call_args
            assert "tools" not in call_args.kwargs

    def test_generate_response_with_tool_definitions(
        self, ai_generator, sample_tool_definitions
    ):
        """Test that tool definitions are passed to Claude"""
        with patch.object(ai_generator.client.messages, "create") as mock_create:
            mock_create.return_value = MockResponse(
                stop_reason="end_turn",
                content=[MockContentBlock("text", text="Response")],
            )

            # Generate response with tools
            ai_generator.generate_response(
                query="Search for ML", tools=sample_tool_definitions, tool_manager=None
            )

            # Verify tools were passed
            call_args = mock_create.call_args
            assert "tools" in call_args.kwargs
            assert call_args.kwargs["tools"] == sample_tool_definitions
            assert "tool_choice" in call_args.kwargs
            assert call_args.kwargs["tool_choice"] == {"type": "auto"}

    def test_generate_response_triggers_tool_execution(
        self, ai_generator, sample_tool_definitions, mock_tool_manager
    ):
        """Test that tool use triggers execution flow"""
        with patch.object(ai_generator.client.messages, "create") as mock_create:
            # First call: Claude requests tool use
            first_response = MockResponse(
                stop_reason="tool_use",
                content=[
                    MockContentBlock("text", text="Let me search for that"),
                    MockContentBlock(
                        "tool_use",
                        tool_use_data={
                            "name": "search_course_content",
                            "id": "tool_123",
                            "input": {"query": "machine learning"},
                        },
                    ),
                ],
            )

            # Second call: Claude responds with final answer
            second_response = MockResponse(
                stop_reason="end_turn",
                content=[
                    MockContentBlock("text", text="Machine learning is a field of AI")
                ],
            )

            mock_create.side_effect = [first_response, second_response]

            # Generate response
            response = ai_generator.generate_response(
                query="What is machine learning?",
                tools=sample_tool_definitions,
                tool_manager=mock_tool_manager,
            )

            # Verify two API calls were made
            assert mock_create.call_count == 2

            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="machine learning"
            )

            # Verify final response
            assert "Machine learning is a field of AI" in response

    def test_tool_execution_with_conversation_history(
        self, ai_generator, sample_tool_definitions, mock_tool_manager
    ):
        """Test that conversation history is included in system prompt"""
        with patch.object(ai_generator.client.messages, "create") as mock_create:
            mock_create.return_value = MockResponse(
                stop_reason="end_turn",
                content=[MockContentBlock("text", text="Response")],
            )

            # Generate with history
            ai_generator.generate_response(
                query="Follow-up question",
                conversation_history="User: Previous question\nAssistant: Previous answer",
                tools=sample_tool_definitions,
                tool_manager=mock_tool_manager,
            )

            # Verify system prompt includes history
            call_args = mock_create.call_args
            system_content = call_args.kwargs["system"]
            assert "Previous conversation:" in system_content
            assert "Previous question" in system_content

    def test_handle_tool_execution_formats_messages_correctly(
        self, ai_generator, mock_tool_manager
    ):
        """Test that _execute_tool_calling_loop formats messages correctly"""
        # Create initial response with tool use
        initial_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock("text", text="Searching..."),
                MockContentBlock(
                    "tool_use",
                    tool_use_data={
                        "name": "search_course_content",
                        "id": "tool_456",
                        "input": {"query": "supervised learning"},
                    },
                ),
            ],
        )

        base_params = {
            "messages": [
                {"role": "user", "content": "Tell me about supervised learning"}
            ],
            "system": "You are a helpful assistant",
            "tools": [{"name": "search_course_content"}],
        }

        with patch.object(ai_generator.client.messages, "create") as mock_create:
            mock_create.return_value = MockResponse(
                stop_reason="end_turn",
                content=[
                    MockContentBlock(
                        "text", text="Supervised learning uses labeled data"
                    )
                ],
            )

            # Execute
            response = ai_generator._execute_tool_calling_loop(
                initial_response, base_params, mock_tool_manager
            )

            # Verify message structure
            final_call = mock_create.call_args
            messages = final_call.kwargs["messages"]

            # Should have 3 messages: original user, assistant with tool use, user with tool result
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"

            # Verify tool result format
            assert isinstance(messages[2]["content"], list)
            assert messages[2]["content"][0]["type"] == "tool_result"
            assert messages[2]["content"][0]["tool_use_id"] == "tool_456"

    def test_multiple_tool_calls_in_one_response(self, ai_generator, mock_tool_manager):
        """Test handling multiple tool uses in a single response"""
        # Create response with multiple tool uses
        initial_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_use_data={
                        "name": "search_course_content",
                        "id": "tool_1",
                        "input": {"query": "machine learning"},
                    },
                ),
                MockContentBlock(
                    "tool_use",
                    tool_use_data={
                        "name": "search_course_content",
                        "id": "tool_2",
                        "input": {"query": "deep learning"},
                    },
                ),
            ],
        )

        base_params = {
            "messages": [{"role": "user", "content": "Compare ML and DL"}],
            "system": "You are a helpful assistant",
            "tools": [{"name": "search_course_content"}],
        }

        with patch.object(ai_generator.client.messages, "create") as mock_create:
            mock_create.return_value = MockResponse(
                stop_reason="end_turn",
                content=[MockContentBlock("text", text="Comparison result")],
            )

            # Execute
            ai_generator._execute_tool_calling_loop(
                initial_response, base_params, mock_tool_manager
            )

            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2

    def test_tool_execution_without_tools_parameter_returns_none(self, ai_generator):
        """Test that tools are included in rounds within limit, excluded after"""
        initial_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_use_data={
                        "name": "search_course_content",
                        "id": "tool_123",
                        "input": {"query": "test"},
                    },
                )
            ],
        )

        base_params = {
            "messages": [{"role": "user", "content": "Test"}],
            "system": "Test system",
            "tools": [{"name": "search_course_content"}],  # Tools in first call
            "tool_choice": {"type": "auto"},
        }

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        with patch.object(ai_generator.client.messages, "create") as mock_create:
            mock_create.return_value = MockResponse(
                stop_reason="end_turn", content=[MockContentBlock("text", text="Final")]
            )

            ai_generator._execute_tool_calling_loop(
                initial_response, base_params, mock_tool_manager
            )

            # Verify the call
            # In this case, round 0 executed the tool, then round 1 makes API call with tools
            # (because current_round=1 < MAX_TOOL_ROUNDS=2), and gets end_turn, so it returns
            final_call = mock_create.call_args
            # Tools should still be included since we're in round 1 (< MAX_TOOL_ROUNDS)
            assert "tools" in final_call.kwargs
            assert "tool_choice" in final_call.kwargs

    def test_system_prompt_contains_tool_instructions(self, ai_generator):
        """Test that system prompt includes tool usage instructions"""
        # The SYSTEM_PROMPT should contain instructions about tool usage
        assert "tool" in ai_generator.SYSTEM_PROMPT.lower()
        assert "search" in ai_generator.SYSTEM_PROMPT.lower()

    def test_temperature_and_max_tokens_settings(self, ai_generator):
        """Test that correct temperature and max_tokens are set"""
        assert ai_generator.base_params["temperature"] == 0
        assert ai_generator.base_params["max_tokens"] == 800
        assert ai_generator.base_params["model"] == "claude-sonnet-4-20250514"

    def test_error_handling_in_tool_execution(self, ai_generator, mock_tool_manager):
        """Test that errors during tool execution are handled"""
        # Setup tool manager to raise exception
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        initial_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_use_data={
                        "name": "search_course_content",
                        "id": "tool_123",
                        "input": {"query": "test"},
                    },
                )
            ],
        )

        base_params = {
            "messages": [{"role": "user", "content": "Test"}],
            "system": "Test",
        }

        # Should raise the exception (not swallow it)
        with pytest.raises(Exception, match="Tool execution failed"):
            ai_generator._execute_tool_calling_loop(
                initial_response, base_params, mock_tool_manager
            )

    def test_two_sequential_tool_calls(
        self, ai_generator, sample_tool_definitions, mock_tool_manager
    ):
        """Test that Claude can make 2 sequential tool calls in separate rounds"""
        with patch.object(ai_generator.client.messages, "create") as mock_create:
            # Round 0: Initial request with first tool use
            round0_response = MockResponse(
                stop_reason="tool_use",
                content=[
                    MockContentBlock("text", text="Let me search for that"),
                    MockContentBlock(
                        "tool_use",
                        tool_use_data={
                            "name": "search_course_content",
                            "id": "tool_1",
                            "input": {"query": "machine learning"},
                        },
                    ),
                ],
            )

            # Round 1: Second tool use after seeing first results
            round1_response = MockResponse(
                stop_reason="tool_use",
                content=[
                    MockContentBlock("text", text="Let me refine that search"),
                    MockContentBlock(
                        "tool_use",
                        tool_use_data={
                            "name": "search_course_content",
                            "id": "tool_2",
                            "input": {"query": "neural networks", "lesson_number": 5},
                        },
                    ),
                ],
            )

            # Final round: End with answer
            final_response = MockResponse(
                stop_reason="end_turn",
                content=[
                    MockContentBlock(
                        "text",
                        text="Machine learning uses neural networks for pattern recognition.",
                    )
                ],
            )

            # First API call already happened in generate_response, so we mock only the loop calls
            mock_create.side_effect = [round1_response, final_response]

            # Execute
            response = ai_generator._execute_tool_calling_loop(
                round0_response,
                {
                    "messages": [
                        {"role": "user", "content": "What is machine learning?"}
                    ],
                    "system": "Test system",
                    "tools": sample_tool_definitions,
                },
                mock_tool_manager,
            )

            # Verify
            assert mock_create.call_count == 2  # Round 1 and final call
            assert mock_tool_manager.execute_tool.call_count == 2  # Two tools executed
            assert "neural networks" in response.lower()

            # Verify message structure in final call
            final_call = mock_create.call_args_list[1]
            messages = final_call.kwargs["messages"]
            # Should have: user, asst1, results1, asst2, results2
            assert len(messages) == 5
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[3]["role"] == "assistant"
            assert messages[4]["role"] == "user"

    def test_max_rounds_enforced(
        self, ai_generator, sample_tool_definitions, mock_tool_manager
    ):
        """Test that after MAX_TOOL_ROUNDS, a final call is made WITHOUT tools"""
        with patch.object(ai_generator.client.messages, "create") as mock_create:
            # Both rounds return tool_use (Claude wants to keep searching)
            tool_use_response = MockResponse(
                stop_reason="tool_use",
                content=[
                    MockContentBlock(
                        "tool_use",
                        tool_use_data={
                            "name": "search_course_content",
                            "id": "tool_x",
                            "input": {"query": "test"},
                        },
                    )
                ],
            )

            final_response = MockResponse(
                stop_reason="end_turn",
                content=[
                    MockContentBlock(
                        "text", text="Here's what I found based on searches."
                    )
                ],
            )

            # Mock returns tool_use twice, forcing max rounds
            mock_create.side_effect = [tool_use_response, final_response]

            # Execute (round 0 is initial_response)
            response = ai_generator._execute_tool_calling_loop(
                tool_use_response,
                {
                    "messages": [{"role": "user", "content": "Test"}],
                    "system": "Test",
                    "tools": sample_tool_definitions,
                },
                mock_tool_manager,
            )

            # Verify 2 API calls (round 1 with tools, round 2 without tools)
            assert mock_create.call_count == 2

            # Verify round 1 has tools
            round1_call = mock_create.call_args_list[0]
            assert "tools" in round1_call.kwargs
            assert round1_call.kwargs["tools"] == sample_tool_definitions

            # Verify round 2 (final call) has NO tools
            final_call = mock_create.call_args_list[1]
            assert "tools" not in final_call.kwargs
            assert "tool_choice" not in final_call.kwargs

    def test_early_termination_after_one_round(
        self, ai_generator, sample_tool_definitions, mock_tool_manager
    ):
        """Test that if Claude finishes after 1 round, loop exits early"""
        with patch.object(ai_generator.client.messages, "create") as mock_create:
            # Initial tool use
            initial_response = MockResponse(
                stop_reason="tool_use",
                content=[
                    MockContentBlock(
                        "tool_use",
                        tool_use_data={
                            "name": "search_course_content",
                            "id": "tool_1",
                            "input": {"query": "test"},
                        },
                    )
                ],
            )

            # After first round, Claude returns end_turn (satisfied with results)
            final_response = MockResponse(
                stop_reason="end_turn",
                content=[
                    MockContentBlock("text", text="Found the answer in first search.")
                ],
            )

            mock_create.return_value = final_response

            # Execute
            response = ai_generator._execute_tool_calling_loop(
                initial_response,
                {
                    "messages": [{"role": "user", "content": "Test"}],
                    "system": "Test",
                    "tools": sample_tool_definitions,
                },
                mock_tool_manager,
            )

            # Verify only 1 API call made (early termination)
            assert mock_create.call_count == 1
            assert mock_tool_manager.execute_tool.call_count == 1
            assert "first search" in response.lower()

    def test_message_history_accumulates_correctly(
        self, ai_generator, sample_tool_definitions, mock_tool_manager
    ):
        """Test that message history accumulates correctly across rounds"""
        with patch.object(ai_generator.client.messages, "create") as mock_create:
            # Initial response with tool use
            initial_response = MockResponse(
                stop_reason="tool_use",
                content=[
                    MockContentBlock("text", text="Searching..."),
                    MockContentBlock(
                        "tool_use",
                        tool_use_data={
                            "name": "search_course_content",
                            "id": "tool_1",
                            "input": {"query": "first search"},
                        },
                    ),
                ],
            )

            # Second tool use
            second_response = MockResponse(
                stop_reason="tool_use",
                content=[
                    MockContentBlock("text", text="Refining..."),
                    MockContentBlock(
                        "tool_use",
                        tool_use_data={
                            "name": "search_course_content",
                            "id": "tool_2",
                            "input": {"query": "second search"},
                        },
                    ),
                ],
            )

            # Final answer
            final_response = MockResponse(
                stop_reason="end_turn",
                content=[MockContentBlock("text", text="Complete answer")],
            )

            mock_create.side_effect = [second_response, final_response]

            # Execute
            ai_generator._execute_tool_calling_loop(
                initial_response,
                {
                    "messages": [{"role": "user", "content": "Original question"}],
                    "system": "Test",
                    "tools": sample_tool_definitions,
                },
                mock_tool_manager,
            )

            # Verify final message structure
            final_call = mock_create.call_args_list[1]
            messages = final_call.kwargs["messages"]

            # Should be: [user, asst1, tool_result1, asst2, tool_result2]
            assert len(messages) == 5
            assert messages[0]["content"] == "Original question"
            assert messages[0]["role"] == "user"

            # Assistant 1 with tool use
            assert messages[1]["role"] == "assistant"
            assert any(block.type == "tool_use" for block in messages[1]["content"])

            # Tool results 1
            assert messages[2]["role"] == "user"
            assert messages[2]["content"][0]["type"] == "tool_result"
            assert messages[2]["content"][0]["tool_use_id"] == "tool_1"

            # Assistant 2 with tool use
            assert messages[3]["role"] == "assistant"

            # Tool results 2
            assert messages[4]["role"] == "user"
            assert messages[4]["content"][0]["tool_use_id"] == "tool_2"

    def test_tool_execution_error_in_loop(self, ai_generator, sample_tool_definitions):
        """Test that tool execution errors propagate and stop the loop"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception(
            "Database connection failed"
        )

        initial_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    "tool_use",
                    tool_use_data={
                        "name": "search_course_content",
                        "id": "tool_1",
                        "input": {"query": "test"},
                    },
                )
            ],
        )

        with patch.object(ai_generator.client.messages, "create") as mock_create:
            # Should raise exception before making any more API calls
            with pytest.raises(Exception, match="Database connection failed"):
                ai_generator._execute_tool_calling_loop(
                    initial_response,
                    {
                        "messages": [{"role": "user", "content": "Test"}],
                        "system": "Test",
                        "tools": sample_tool_definitions,
                    },
                    mock_tool_manager,
                )

            # No API calls should have been made (error happened during tool execution)
            assert mock_create.call_count == 0
