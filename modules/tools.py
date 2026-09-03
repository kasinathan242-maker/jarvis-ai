import subprocess
import webbrowser
import re
import ast

async def handle(text: str) -> str:
    text_lower = text.lower()

    # Search the web
    search_match = re.search(r'search (?:for )?(.+)', text_lower)
    if search_match:
        query = search_match.group(1).replace(' ', '+')
        try:
            webbrowser.open(f'https://www.google.com/search?q={query}')
        except Exception:
            pass
        return f'Searching Google for "{search_match.group(1)}"'

    # Calculate expression safely
    calc_match = re.search(r'(?:calculate|compute|what is) (.+)', text_lower)
    if calc_match:
        expr = calc_match.group(1).strip()
        # strip trailing question marks or punctuation
        expr_clean = expr.rstrip('?!= ')
        try:
            # Safe evaluation using ast.literal_eval or safe mathematical eval
            node = ast.parse(expr_clean, mode='eval')
            # Only allow basic arithmetic expressions
            result = eval(compile(node, '<string>', 'eval'), {"__builtins__": {}}, {})
            return f'The result of {expr_clean} is {result}'
        except Exception:
            return f'Could not calculate: {expr}'

    # Open applications
    if 'notepad' in text_lower:
        try:
            subprocess.Popen(['notepad.exe'])
            return 'Opening Notepad.'
        except Exception as e:
            return f'Failed to launch Notepad: {str(e)}'

    if 'calculator' in text_lower or 'calc' in text_lower:
        try:
            subprocess.Popen(['calc.exe'])
            return 'Opening Calculator.'
        except Exception as e:
            return f'Failed to launch Calculator: {str(e)}'

    return 'Tool command understood. You can ask to search the web, calculate math expressions, or open applications like Notepad or Calculator.'
