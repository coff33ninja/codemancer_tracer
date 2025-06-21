import networkx as nx
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from .parser import parse_ast, extract_info

def build_call_graph(files: List[Path]) -> Tuple[nx.DiGraph, str, Dict[str, List[str]], Dict]:
    """Builds a NetworkX graph and summary data from parsed files."""
    G = nx.DiGraph()
    summary_md = []
    summary_json = {}
    func_map = defaultdict(list)

    for filepath in files:
        tree = parse_ast(filepath)
        if not tree:
            continue

        functions, imports, calls = extract_info(tree)
        module = filepath.stem
        summary_md.append(f"## {module}\n")
        summary_md.append("**Functions:** " + ", ".join(functions or ["None"]) + "\n")
        summary_md.append("**Imports:** " + ", ".join(set(imports) or ["None"]) + "\n")
        summary_json[module] = {
            "functions": functions or [],
            "imports": sorted(list(set(imports))) or [],
            "calls": calls
        }

        for f in functions:
            G.add_node(f"{module}.{f}", module=module)
            func_map[module].append(f"{module}.{f}")

        # Add edges from functions to the calls they make
        for func_signature, func_calls in calls.items():
            if func_signature != "__module__":
                source_node = f"{module}.{func_signature}"
                for call in func_calls:
                    G.add_edge(source_node, call)

    return G, "\n".join(summary_md), func_map, summary_json

def create_pyvis_graph(G: nx.DiGraph, func_map: Dict[str, List[str]], out_file: str = "function_map.html"):
    """Generates an interactive Pyvis graph HTML file."""
    from pyvis.network import Network # Import inside function for lazy loading
    # Define network options, including the interactive filter for groups
    options = {
        "directed": True,
        "height": "800px",
        "width": "100%",
        "bgcolor": "#222222",
        "font_color": "white", # This is a direct argument to Network, and is passed via **options.
        "notebook": False # Important for standalone HTML generation.
        # Removed "configure" as it caused TypeError in __init__
    }
    net = Network(**options) # Pass the options dictionary to the constructor
    color_map = {}
    # Expanded color palette for more modules
    colors = [
        "#FF6666", "#66FF66", "#6699FF", "#FFFF66", "#FF66FF", "#66FFFF",
        "#FF9933", "#99FF33", "#3399FF", "#FF33FF", "#33FFFF", "#FFCC00",
        "#CCFF00", "#00FFCC", "#00CCFF", "#CC00FF", "#FF00CC", "#FF9999",
        "#99FF99", "#9999FF", "#FFFF99", "#FF99FF", "#99FFFF", "#CCCCCC"
    ]
    for i, module in enumerate(func_map):
        color_map[module] = colors[i % len(colors)]

    for node, data in G.nodes(data=True):
        mod = data.get("module", "Other")
        # Extract just the function signature part from the node name
        # If it's an external call (mod == "Other"), the node name is already just the call name
        function_signature_only = node.replace(f"{mod}.", "") if node.startswith(f"{mod}.") else node
        # Create the new label with a line break and smaller font for the module name, and set the 'group' for the legend
        new_label = f"{function_signature_only}<br><span style='font-size: 0.8em;'>({mod})</span>"
        net.add_node(node, label=new_label, color=color_map.get(mod, "#FFFFFF"), title=f"Module: {mod}", group=mod)

    for src, dst in G.edges():
        net.add_edge(src, dst)

    # Enable the interactive filter UI, which creates a legend based on the 'group' property.
    net.show_buttons() # Call without arguments to enable default configuration options.
    # The following line actually saves the HTML file to disk.
    net.show(out_file, notebook=False)
    print(f"[+] Interactive map saved as {out_file}")
