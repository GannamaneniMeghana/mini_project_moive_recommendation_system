
import os

path = r"d:\Mini_project_movie_\static\style.css"

with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# We want to keep everything up to the first closing of .remember-me which appears to be around line 1306
# Let's find the line index for the first ".remember-me" block's closing brace.
# We know the duplicate starts after that.

final_lines = []
capturing = True
remember_me_found = False

for i, line in enumerate(lines):
    if ".remember-me" in line:
        remember_me_found = True
    
    final_lines.append(line)
    
    # If we are inside the first remember-me block and hit the closing brace
    if remember_me_found and line.strip() == "}":
        # Check if this is indeed the end of the file (or should be)
        # The duplicated content starts with "/* Results */" or "overflow-x: auto" (from the bad replace)
        # Let's just stop here.
        break

# Write back
with open(path, "w", encoding="utf-8") as f:
    f.writelines(final_lines)

print(f"Truncated file to {len(final_lines)} lines.")
