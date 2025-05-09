import tiktoken

def calculate_token_length(input_string: str) -> int:
    """
    Calculate the token length of a given string using tiktoken.

    Args:
        input_string (str): The input string to calculate token length for.

    Returns:
        int: The length of the tokens.
    """
    # Initialize the tokenizer (you can specify a model if needed, e.g., 'gpt-4')
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Encode the input string to tokens
    tokens = tokenizer.encode(input_string)
    
    # Return the length of the tokens
    return len(tokens)