import networkx as nx
import numpy as np
from typing import Dict, Tuple, Any, List, Optional, Set


class MarkovTree:
    """
    An explicit Markov tree implementation that supports negative probabilities
    for retrospective inference.
    
    This class uses a directed graph to represent states and transitions,
    allowing for negative probability values to represent retrospective reasoning.
    """
    
    def __init__(self):
        """Initialize an empty Markov tree."""
        # Use DiGraph for directed transitions between states
        self.graph = nx.DiGraph()
        # Track visited states for faster lookup
        self.visited_states: Set[Any] = set()
    
    def add_state(self, state) -> None:
        """
        Add a state to the Markov tree.
        
        Args:
            state: A hashable representation of the state
        """
        if state not in self.visited_states:
            self.graph.add_node(state)
            self.visited_states.add(state)
    
    def add_transition(self, from_state, to_state, action, probability: float) -> None:
        """
        Add a transition between states with an associated action and probability.
        
        Args:
            from_state: Source state
            to_state: Destination state
            action: The action that causes the transition
            probability: Transition probability (can be negative)
        """
        # Ensure both states exist in the graph
        self.add_state(from_state)
        self.add_state(to_state)
        
        # Add the transition with the associated probability and action
        self.graph.add_edge(
            from_state, 
            to_state, 
            action=action, 
            probability=probability
        )
    
    def get_probability(self, from_state, to_state, action=None) -> float:
        """
        Get the probability of a transition between states.
        
        Args:
            from_state: Source state
            to_state: Destination state
            action: The action that causes the transition (optional)
            
        Returns:
            Transition probability (can be negative)
        """
        if not self.graph.has_edge(from_state, to_state):
            return 0.0
        
        edge_data = self.graph.get_edge_data(from_state, to_state)
        
        # If action is specified, check if it matches
        if action is not None and edge_data['action'] != action:
            return 0.0
            
        return edge_data['probability']
    
    def set_probability(self, from_state, to_state, probability: float, action=None) -> None:
        """
        Set or update the probability of a transition.
        
        Args:
            from_state: Source state
            to_state: Destination state
            probability: New transition probability (can be negative)
            action: The action to update (if None, updates the existing action)
        """
        if not self.graph.has_edge(from_state, to_state):
            raise ValueError(f"Transition from {from_state} to {to_state} does not exist")
        
        edge_data = self.graph.get_edge_data(from_state, to_state)
        edge_action = edge_data['action']
        
        # If action is specified, ensure it matches existing action
        if action is not None and action != edge_action:
            raise ValueError(f"Action mismatch: edge has action {edge_action}, requested {action}")
            
        # Update the probability
        self.graph[from_state][to_state]['probability'] = probability
    
    def get_outgoing_transitions(self, state) -> List[Tuple[Any, Any, float]]:
        """
        Get all possible transitions from a state.
        
        Args:
            state: The source state
            
        Returns:
            List of (destination_state, action, probability) tuples
        """
        if state not in self.graph:
            return []
            
        transitions = []
        for successor in self.graph.successors(state):
            edge_data = self.graph.get_edge_data(state, successor)
            transitions.append((successor, edge_data['action'], edge_data['probability']))
            
        return transitions
    
    def get_incoming_transitions(self, state) -> List[Tuple[Any, Any, float]]:
        """
        Get all transitions leading to a state.
        
        Args:
            state: The destination state
            
        Returns:
            List of (source_state, action, probability) tuples
        """
        if state not in self.graph:
            return []
            
        transitions = []
        for predecessor in self.graph.predecessors(state):
            edge_data = self.graph.get_edge_data(predecessor, state)
            transitions.append((predecessor, edge_data['action'], edge_data['probability']))
            
        return transitions
    
    def backward_inference(self, current_state, depth: int = 1) -> Dict[Any, float]:
        """
        Perform backward inference to infer probabilities of previous states.
        
        This is where negative probabilities become useful for retrospective reasoning.
        
        Args:
            current_state: The current state from which to reason backwards
            depth: How many steps back to perform inference
            
        Returns:
            Dictionary mapping potential previous states to their probabilities
        """
        if depth <= 0 or current_state not in self.graph:
            return {}
            
        previous_states = {}
        
        # First level: direct predecessors
        for predecessor in self.graph.predecessors(current_state):
            edge_data = self.graph.get_edge_data(predecessor, current_state)
            previous_states[predecessor] = edge_data['probability']
            
        # If depth > 1, continue recursively
        if depth > 1:
            deeper_states = {}
            for state, prob in previous_states.items():
                # Recursively get earlier states
                earlier_states = self.backward_inference(state, depth - 1)
                
                # Combine probabilities
                for earlier_state, earlier_prob in earlier_states.items():
                    combined_prob = prob * earlier_prob
                    if earlier_state in deeper_states:
                        deeper_states[earlier_state] += combined_prob
                    else:
                        deeper_states[earlier_state] = combined_prob
                        
            # Update with deeper states
            for state, prob in deeper_states.items():
                if state in previous_states:
                    previous_states[state] += prob
                else:
                    previous_states[state] = prob
                    
        return previous_states
    
    def visualize(self) -> nx.DiGraph:
        """
        Return a visualization-ready graph.
        
        Returns:
            A NetworkX DiGraph that can be visualized
        """
        return self.graph
