"""Writes content to a file and returns the absolute path."""

import os


def write_file(file_path: str, content: str, create_dirs: bool = True, max_size: int = 10 * 1024 * 1024) -> str:
    """Writes content to a file.

    This function performs the following steps:
    1. Expands the tilde (~) in the file path to the user's home directory.
    2. Converts a relative path to an absolute path.
    3. Creates parent directories if they don't exist (optional).
    4. Checks the content size before writing to ensure it is not too large.
    5. Writes the content to the file.
    6. Handles common file operation errors.

    Parameters:
    file_path (str): The path to the file to write to.
    content (str): The content to write to the file.
    create_dirs (bool): Whether to create parent directories if they don't exist.
    max_size (int): Maximum allowed content size in bytes.

    Returns:
    str: The absolute path of the written file.

    Raises:
    ValueError: If the content size exceeds the maximum allowed size.
    PermissionError: If the file cannot be written due to permission issues.
    OSError: If other OS-related errors occur.
    """
    try:
        # Expand tilde to user's home directory
        expanded_path = os.path.expanduser(file_path)

        # Convert relative path to absolute path
        absolute_path = os.path.abspath(expanded_path)

        # Check content size
        content_size = len(content.encode('utf-8'))
        if content_size > max_size:
            raise ValueError(f"Content size ({content_size} bytes) exceeds the maximum allowed size ({max_size} bytes).")

        # Create parent directories if needed
        if create_dirs:
            os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

        # Write the content to the file
        with open(absolute_path, 'w', encoding='utf-8') as file:
            file.write(content)

        return absolute_path

    except PermissionError:
        raise PermissionError(f"Permission denied: Unable to write to the file '{absolute_path}'.")
    except OSError as e:
        raise OSError(f"An error occurred while writing the file: {e}")
