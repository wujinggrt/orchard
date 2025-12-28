from openai import OpenAI, AsyncOpenAI, APIError
from openai.types.chat import ChatCompletion
from typing import Union, Any
from orchard.llm.schema import Message, ROLE_VALUES
from orchard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        if not self.api_key or not self.base_url or not self.model:
            raise ValueError("API key, base URL, and model must be provided")
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    @staticmethod
    def format_messages(
        *, messages: list[Union[dict, Message]], supports_images: bool = False
    ) -> list[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                                # "url": f"data:image;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    def get_chat_completion(
        self,
        *,
        messages: list[Union[dict, Message]],
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ChatCompletion:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            temperature (float): Sampling temperature for the response
            tools (list[dict[str, Any]]): List of tools to use in the response

        Returns:
            str: The generated response
        """
        # For ask_with_images, we always set supports_images to True because
        # this method should only be called with models that support images
        # Format messages with image support
        try:
            formatted_messages = self.format_messages(
                messages=messages, supports_images=True
            )
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temperature,
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            response = self.client.chat.completions.create(**params)
            return response
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def get_chat_completion_async(
        self,
        *,
        messages: list[Union[dict, Message]],
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> ChatCompletion:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response
        """
        try:
            formatted_messages = self.format_messages(
                messages=messages, supports_images=True
            )
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": temperature,
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
            response = await self.async_client.chat.completions.create(**params)
            return response
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
