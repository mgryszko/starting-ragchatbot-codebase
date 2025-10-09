from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- Use search tool **only** for questions about specific course content or detailed educational materials
- Use outline tool for questions about course structure, lesson list, or course overview
- **You may use tools up to 2 times per conversation** to gather comprehensive information
- Use tools strategically:
  * Round 1: Initial broad search or course outline retrieval
  * Round 2: Refined search with specific filters (course/lesson) if first results suggest more is needed
- Examples of effective multi-round usage:
  * Broad topic search → Specific lesson search
  * Course outline retrieval → Specific lesson content search
  * One course search → Related course search for comparison
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use the outline tool to retrieve course title, link, instructor, and complete lesson list
- **Course content questions**: Use search tool (once or twice as needed), then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool usage explanations, or question-type analysis
 - Do not mention "based on the search results", "I searched multiple times", or "based on the outline"
 - Integrate information from multiple searches seamlessly

For course outline queries:
- Always include the complete course title and course link
- List all lessons with their numbers and titles
- Include instructor name when available

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Maximum number of sequential tool calling rounds per query
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed (supports up to MAX_TOOL_ROUNDS)
        if response.stop_reason == "tool_use" and tool_manager:
            return self._execute_tool_calling_loop(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _execute_tool_calling_loop(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Execute tool calling loop with up to MAX_TOOL_ROUNDS sequential rounds.

        Args:
            initial_response: The response containing initial tool use requests
            base_params: Base API parameters (messages, system, tools, etc.)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after all tool rounds complete
        """
        # Initialize state
        messages = base_params["messages"].copy()
        current_round = 0
        response = initial_response

        # Loop for up to MAX_TOOL_ROUNDS
        while current_round < self.MAX_TOOL_ROUNDS:
            # Check termination: if not tool_use, we're done
            if response.stop_reason != "tool_use":
                return self._extract_text_from_response(response)

            # Append assistant response with tool uses
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tools for this round
            tool_results = self._execute_tools(response, tool_manager)

            # Append tool results
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Increment round counter
            current_round += 1

            # Prepare next API call
            if current_round < self.MAX_TOOL_ROUNDS:
                # Still within limit - include tools for potential next round
                api_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"],
                    "tools": base_params.get("tools"),
                    "tool_choice": {"type": "auto"},
                }
            else:
                # Max rounds reached - final call WITHOUT tools
                api_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"],
                }

            # Make next API call
            response = self.client.messages.create(**api_params)

        # Exited loop - return final response
        return self._extract_text_from_response(response)

    def _execute_tools(self, response, tool_manager) -> List[Dict[str, Any]]:
        """
        Execute all tool_use blocks in the response and return formatted results.

        Args:
            response: API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool_result dictionaries
        """
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                # Execute the tool
                result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                # Format result for API
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result,
                    }
                )

        return tool_results

    def _extract_text_from_response(self, response) -> str:
        """
        Extract text content from API response.

        Args:
            response: API response object

        Returns:
            Extracted text, or empty string if no text blocks found
        """
        for content_block in response.content:
            if content_block.type == "text":
                return content_block.text
        return ""
