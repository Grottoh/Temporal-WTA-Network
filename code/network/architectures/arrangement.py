from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Optional

#from ...utils.util import Constants as C
from ..component import Component

@dataclass
class Node:

    position: tuple[int, int]
    component: Component

class Arrangement:

    def __init__(
            self, 
            nodes: list[Node] = [],
            ):
        
        # Add each node to the network
        self.nodes: list[Node] = []
        for node in nodes:
            self.add_node(node)
    
    @property
    def positions(self) -> list[ tuple[int, int] ]:
        """ Return a list of the node positions. """
        return [node.position for node in self.nodes]

    @property
    def roots(self) -> list[Component]:
        """ Return a list of the components of the root nodes. """
        return [node.component for node in self.nodes if node.position[0] == 0]
    
    @property
    def components(self) -> list[Component]:
        """ Return a list of all the node components. """
        return [node.component for node in self.nodes]
    
    def comp2node(self, component: Component) -> Node:
        """ Return the node corresponding to the given component. """
        for node in self.nodes:
            if node.component == component:
                return node
        raise ValueError( "None of the nodes correspond to component" +
                         f"<{component}>.")
    
    def add_node(self, node: Node):
        """ Add a node to the network arrangement. """

        # Ensure no two nodes occupy the same position
        if node.position in self.positions:
            raise ValueError(f"Position {node.position} is already occupied.")
        elif node.component in self.components:
            raise ValueError(f"Component {node.component} is" + 
                                "assigned to multiple nodes.")
        
        # Keep track of the node in various manners
        self.nodes.append(node)

        # Ensure the nodes are sorted by their position
        self.sort()
    
    def remove_node(self, component: Component):
        """ Remove the node associated with the given component. """
        try:
            idx_component = self.components.index(component)
            self.nodes.pop(idx_component)
        except:
            raise ValueError(f"Component <{component}> is not present" +
                              " in the network arrangement.")
    
    def sort(self):
        """ Sort all nodes by position. """
        self.nodes.sort(key=lambda node: node.position)
    


            




