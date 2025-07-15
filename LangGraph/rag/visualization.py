from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph


def visualize_graph(graph: CompiledStateGraph) -> None:
    display(Image(graph.get_graph().draw_mermaid_png()))
