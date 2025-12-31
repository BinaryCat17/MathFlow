import json
import os
import glob

def format_json_mathflow(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        output = "{\n"
        keys = list(data.keys())
        
        for k_idx, key in enumerate(keys):
            val = data[key]
            output += f'    "{key}": '
            
            if key in ["nodes", "links"] and isinstance(val, list):
                output += "[\n"
                for i, item in enumerate(val):
                    # SECURE one-line format using separators
                    line = json.dumps(item, ensure_ascii=False, separators=(', ', ': '))
                    # Add padding inside root braces
                    if line.startswith('{') and line.endswith('}'):
                        line = '{ ' + line[1:-1] + ' }'
                    
                    output += f'        {line}'
                    if i < len(val) - 1:
                        output += ","
                    output += "\n"
                output += "    ]"
            else:
                val_str = json.dumps(val, indent=4, ensure_ascii=False)
                indented = val_str.replace('\n', '\n    ')
                output += indented
                
            if k_idx < len(keys) - 1:
                output += ","
            output += "\n"
            
        output += "}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
            f.write('\n')
            
        print(f"Formatted {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    files = []
    for pattern in ['**/*.json', '**/*.mfapp']:
        files.extend(glob.glob(pattern, recursive=True))
    files = [f for f in files if not any(x in f for x in ['build/', 'out/', 'vcpkg', '.git/'])]
    for f in files:
        format_json_mathflow(f)