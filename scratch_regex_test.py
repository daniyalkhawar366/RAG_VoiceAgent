import re

pattern = re.compile(
    r'(?<!\d)[\.!?](?:\s+|\n|$)|'       # Sentence endings (. ! ?)
    r',(?!\d)(?:\s+|\n|$)|'            # Commas with spaces (not thousands)
    r';(?:\s+|\n|$)|'                   # Semicolons
    r'(?:\s+—\s+|\s+-\s+)'             # Dashes/Hyphens with spaces
)

text = "Our cheapest Genesis CPO car is the G80 EV Advance, priced at 99,000 SAR."
buffer = text
while True:
    match = pattern.search(buffer)
    if match:
        idx = match.end()
        sentence = buffer[:idx].strip()
        buffer = buffer[idx:]
        print(f"SPLIT CLAUSE: '{sentence}'")
    else:
        break
if buffer.strip():
    print(f"REMAINING: '{buffer.strip()}'")
