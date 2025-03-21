"""
3D network visualization for X Network Visualization.
"""
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

import streamlit as st
import streamlit.components.v1 as components

from config import DEFAULT_BASE_SIZE, DEFAULT_ACCOUNT_SIZE_FACTOR, DEFAULT_LABEL_SIZE_FACTOR, DEFAULT_NODE_SPACING

logger = logging.getLogger(__name__)

class Network3DVisualizer:
    """
    3D network visualization using Force-Graph.
    """
    
    def __init__(self, size_factors: Optional[Dict] = None):
        """
        Initialize the visualizer.
        
        Args:
            size_factors: Dictionary with size configuration
        """
        self.size_factors = size_factors or {
            'base_size': DEFAULT_BASE_SIZE,
            'importance_factor': DEFAULT_ACCOUNT_SIZE_FACTOR,
            'label_size_factor': DEFAULT_LABEL_SIZE_FACTOR,
            'node_spacing': DEFAULT_NODE_SPACING
        }
    
    def build_visualization(self, nodes: Dict[str, Dict], 
                           edges: List[Tuple[str, str]], 
                           selected_nodes: List[str],
                           importance_scores: Dict[str, float],
                           cloutrank_scores: Dict[str, float],
                           in_degrees: Dict[str, int],
                           use_pagerank: bool = False) -> str:
        """
        Constructs a 3D ForceGraph visualization with permanent labels and hover info.
        
        Args:
            nodes: Dictionary mapping node IDs to node data
            edges: List of (source, target) edge tuples
            selected_nodes: List of node IDs to include in visualization
            importance_scores: Dictionary mapping node IDs to importance scores
            cloutrank_scores: Dictionary mapping node IDs to CloutRank scores
            in_degrees: Dictionary mapping node IDs to in-degree values
            use_pagerank: Whether to use PageRank for node sizing
            
        Returns:
            HTML code for the visualization
        """
        # Filter nodes and edges to only include selected nodes
        filtered_nodes = {node_id: meta for node_id, meta in nodes.items() if node_id in selected_nodes}
        
        # Ensure that we only include edges where both source and target exist in filtered_nodes
        filtered_edges = []
        for src, tgt in edges:
            if src in filtered_nodes and tgt in filtered_nodes:
                filtered_edges.append((src, tgt))
                
        # Debug information
        logger.info(f"Building visualization with {len(filtered_nodes)} nodes and {len(filtered_edges)} edges")
        
        # Additional debugging for edges
        original_id = next((id for id in nodes.keys() if id.startswith("orig_")), None)
        if original_id:
            original_selected = original_id in selected_nodes
            original_filtered = original_id in filtered_nodes
            
            # Count first degree connections
            first_degree_edges_input = [(src, tgt) for src, tgt in edges if src == original_id or tgt == original_id]
            first_degree_edges_filtered = [(src, tgt) for src, tgt in filtered_edges if src == original_id or tgt == original_id]
            
            logger.info(f"Original ID {original_id} selected: {original_selected}, in filtered nodes: {original_filtered}")
            logger.info(f"First degree edges in input: {len(first_degree_edges_input)}, in filtered: {len(first_degree_edges_filtered)}")
            
            for i, (src, tgt) in enumerate(first_degree_edges_filtered[:5]):
                src_name = nodes[src]["screen_name"] if src in nodes else "Unknown"
                tgt_name = nodes[tgt]["screen_name"] if tgt in nodes else "Unknown"
                logger.info(f"  Edge {i+1}: {src} (@{src_name}) → {tgt} (@{tgt_name})")

        nodes_data = []
        links_data = []

        # Convert edges to proper format
        links_data = []
        for src, tgt in filtered_edges:
            links_data.append({"source": str(src), "target": str(tgt)})

        # Convert nodes to proper format with additional info
        for node_id, meta in filtered_nodes.items():
            try:
                base_size = float(self.size_factors.get('base_size', DEFAULT_BASE_SIZE))
                importance_factor = float(self.size_factors.get('importance_factor', DEFAULT_ACCOUNT_SIZE_FACTOR))
                
                # Determine if this is a topic node
                is_topic = meta.get("is_topic", False)
                
                # Handle None values for followers_count
                followers_count = meta.get("followers_count")
                if followers_count is None:
                    followers_count = 0 if node_id.startswith("orig_") else 1000  # Default value for non-original nodes
                
                # Calculate node size with type checking
                if is_topic:
                    # For topic nodes, use topic_influence for sizing
                    topic_influence = meta.get("topic_influence", 1.0)
                    node_size = base_size * importance_factor * 1.5  # Make topics a bit larger
                else:
                    # For regular account nodes
                    followers_factor = float(followers_count) / 1000.0
                    node_size = base_size * importance_factor
                
                # Ensure node_size is positive
                node_size = max(1.0, node_size)
                
                # Handle None values for other metrics
                following_count = meta.get("friends_count", 0)
                if following_count is None:
                    following_count = 0
                    
                ratio = meta.get("ratio", 0.0)
                if ratio is None:
                    ratio = 0.0
                    
                # Get community color if it exists
                community_color = meta.get("community_color", "#6ca6cd")  # Use default color if no community
                
                # Get community information
                username = meta.get("screen_name", "")
                community = "N/A"
                
                if not is_topic:
                    if 'node_communities' in st.session_state and st.session_state.node_communities:
                        if username in st.session_state.node_communities:
                            community_id = st.session_state.node_communities[username]
                            if 'community_labels' in st.session_state and st.session_state.community_labels:
                                community = st.session_state.community_labels.get(community_id, "N/A")
                else:
                    community = "Topic Node"
                
                # Store different data depending on node type
                if is_topic:
                    nodes_data.append({
                        "id": str(node_id),
                        "name": str(meta.get("screen_name", "")),
                        "community": community,
                        "followers": 0,
                        "following": 0,
                        "ratio": 0.0,
                        "size": float(node_size),
                        "description": str(meta.get("description", "")),
                        "cloutrank": 0.0,
                        "indegree": 0,
                        "importance": float(meta.get("topic_influence", 0)),
                        "color": community_color,
                        "isOriginal": False,
                        "isTopic": True,
                        "topic_accounts": len(meta.get("connected_accounts", []))
                    })
                else:
                    # Store actual CloutRank and In-Degree values separately
                    nodes_data.append({
                        "id": str(node_id),
                        "name": str(meta.get("screen_name", "")),
                        "community": community,
                        "followers": int(followers_count),
                        "following": int(following_count),
                        "ratio": float(ratio),
                        "size": float(node_size),
                        "description": str(meta.get("description", "")),
                        "cloutrank": float(cloutrank_scores.get(node_id, 0)),
                        "indegree": int(in_degrees.get(node_id, 0)),
                        "importance": float(importance_scores.get(node_id, 0)),
                        "color": community_color,
                        "isOriginal": bool(node_id.startswith("orig_")),
                        "isTopic": False
                    })
            except Exception as e:
                logger.warning(f"Error processing node {node_id}: {str(e)}")
                is_topic = node_id.startswith("topic_")
                nodes_data.append({
                    "id": str(node_id),
                    "name": str(meta.get("screen_name", "")),
                    "community": "Topic Node" if is_topic else "N/A",
                    "followers": 0,
                    "following": 0,
                    "ratio": 0.0,
                    "size": float(self.size_factors.get('base_size', DEFAULT_BASE_SIZE)),
                    "description": "",
                    "cloutrank": 0.0,
                    "indegree": 0,
                    "importance": 0.0,
                    "color": "#FFA500" if is_topic else "#6ca6cd",
                    "isOriginal": False,
                    "isTopic": is_topic
                })

        # Convert to JSON
        nodes_json = json.dumps(nodes_data)
        links_json = json.dumps(links_data)

        # Update tooltip to correctly show both metrics
        importance_label = "CloutRank" if use_pagerank else "In-Degree"

        # Generate HTML
        html_code = f"""
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8">
            <script src="https://unpkg.com/three@0.149.0/build/three.min.js"></script>
            <script src="https://unpkg.com/3d-force-graph@1.70.10/dist/3d-force-graph.min.js"></script>
            <script src="https://unpkg.com/three-spritetext@1.6.5/dist/three-spritetext.min.js"></script>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
              #graph {{ width: 100%; height: 750px; }}
              .node-tooltip {{
                  font-family: Arial;
                  padding: 8px;
                  border-radius: 4px;
                  background-color: rgba(0,0,0,0.8);
                  color: white;
                  white-space: pre-line;
                  font-size: 14px;
              }}
            </style>
          </head>
          <body>
            <div id="graph"></div>
            <script>
              const data = {{
                nodes: {nodes_json},
                links: {links_json}
              }};
              
              console.log("Graph data:", data);
              
              const Graph = ForceGraph3D()
                (document.getElementById('graph'))
                .graphData(data)
                .nodeColor(node => node.color)
                .nodeRelSize(6)
                .nodeVal(node => node.size)
                .cooldownTime(30000)  // Longer cooldown to ensure stable layout
                .warmupTicks(400)    // More warmup ticks for better initial positioning
                .linkWidth(0.5)      // Thinner links
                .linkOpacity(0.6)    // Semi-transparent links
                .nodeThreeObject(node => {{
                    const group = new THREE.Group();
                    
                    // Use different shapes for different node types
                    let geometry, material;
                    
                    if (node.isTopic) {{
                        // Use a diamond shape for topic nodes
                        const tetrahedronSize = Math.cbrt(node.size) * 1.5;
                        geometry = new THREE.TetrahedronGeometry(tetrahedronSize);
                        material = new THREE.MeshLambertMaterial({{
                            color: node.color,
                            transparent: true,
                            opacity: 0.9,
                            emissive: node.color,
                            emissiveIntensity: 0.3
                        }});
                    }} else if (node.isOriginal) {{
                        // Cube for original account
                        const size = Math.cbrt(node.size);
                        geometry = new THREE.BoxGeometry(size * 1.2, size * 1.2, size * 1.2);
                        material = new THREE.MeshLambertMaterial({{
                            color: node.color,
                            transparent: true,
                            opacity: 0.8
                        }});
                    }} else {{
                        // Sphere for regular accounts
                        geometry = new THREE.SphereGeometry(Math.cbrt(node.size));
                        material = new THREE.MeshLambertMaterial({{
                            color: node.color,
                            transparent: true,
                            opacity: 0.75
                        }});
                    }}
                    
                    const mesh = new THREE.Mesh(geometry, material);
                    group.add(mesh);
                    
                    // Add text label
                    const sprite = new SpriteText(node.name);
                    
                    // Topic nodes get larger, bolder text
                    if (node.isTopic) {{
                        sprite.textHeight = Math.max(5, Math.min(14, 10 * Math.cbrt(node.size / 10))) * {self.size_factors.get('label_size_factor', 1.0)};
                        sprite.fontWeight = 'bold';
                    }} else {{
                        sprite.textHeight = Math.max(4, Math.min(12, 8 * Math.cbrt(node.size / 10))) * {self.size_factors.get('label_size_factor', 1.0)};
                    }}
                                      
                    sprite.color = 'white';
                    sprite.backgroundColor = node.isTopic ? 'rgba(255,165,0,0.7)' : 'rgba(0,0,0,0.6)';
                    sprite.padding = 2;
                    sprite.borderRadius = 3;
                    sprite.position.y = Math.cbrt(node.size) + 1;
                    group.add(sprite);
                    
                    return group;
                }})
                .nodeLabel(node => {{
                    if (node.isTopic) {{
                        return `<div class="node-tooltip">
                            <b>Topic: ${{node.name}}</b><br/>
                            Type: Topic Node<br/>
                            Influence Score: ${{node.importance.toFixed(4)}}<br/>
                            Description: ${{node.description}}
                            </div>`;
                    }} else {{
                        return `<div class="node-tooltip">
                            <b>@${{node.name}}</b><br/>
                            Community: ${{node.community}}<br/>
                            Followers: ${{node.followers.toLocaleString()}}<br/>
                            Following: ${{node.following.toLocaleString()}}<br/>
                            Ratio: ${{node.ratio.toFixed(2)}}<br/>
                            Description: ${{node.description}}<br/>
                            ${importance_label}: ${{node.importance.toFixed(4)}}<br/>
                            CloutRank: ${{node.cloutrank.toFixed(4)}}
                            </div>`;
                    }}
                }})
                .linkDirectionalParticles(1)
                .linkDirectionalParticleSpeed(0.006)
                .linkDirectionalArrowLength(3.5)  // Length of the arrow
                .linkDirectionalArrowRelPos(1)    // Position along the line (1 = end of line)
                .linkDirectionalArrowColor(() => '#ffffff')  // White arrows
                .linkCurvature(0.1)  // Slight curve to better see direction
                .backgroundColor("#101020");

              // Set initial camera position
              Graph.cameraPosition({{ x: 150, y: 150, z: 150 }});

              // Adjust force parameters for better layout
              Graph.d3Force('charge').strength(-500);
              Graph.d3Force('link').distance(link => 80);
              
              // Increase space between nodes
              Graph.d3Force('collision', d3.forceCollide().radius(node => Math.cbrt(node.size) * {self.size_factors.get('node_spacing', DEFAULT_NODE_SPACING)}));
              
              // Add centering force
              Graph.d3Force('center', d3.forceCenter(0, 0, 0).strength(0.05));

              // Add node click behavior for camera focus
              Graph.onNodeClick(node => {{
                  const distance = 40;
                  const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
                  Graph.cameraPosition(
                      {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
                      node,
                      2000
                  );
              }});
            </script>
          </body>
        </html>
        """
        return html_code
    
    def render(self, html_code: str, height: int = 750) -> None:
        """
        Render the visualization in Streamlit.
        
        Args:
            html_code: HTML code for the visualization
            height: Height in pixels
        """
        components.html(html_code, height=height, width=1200)