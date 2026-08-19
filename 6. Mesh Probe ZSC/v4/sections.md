1. load_direction(face_df, edge_df)
   → unit vector along load axis
   → used to project all interior nodes

2. classify_surface_interior(node_df, verts_mm)
   → surface nodes: matched to STL vertices
   → interior nodes: everything else
   → each interior node gets: distance from nearest surface Pi feature,
     projection coordinate along load axis

3. apply_saint_venant_depth(interior_nodes, h_edge_min, sv_factor)
   → nodes within depth: Pi gradient zone
   → nodes beyond depth: Ε cylinder zone
   → boundary nodes at exact depth: Β (2-2-1 sequence applied here)

4. seed_pi_from_center(pi_zone_nodes, load_dir)
   → project onto load axis
   → apply net-3 weight from center outward (5 forward −2 back)
   → center node is seed point

5. apply_cylinder_e_pattern(e_zone_nodes, load_dir)
   → group nodes into lines along load axis by transverse proximity
   → 4 nodes minimum per line
   → active lines at positions 1, 3, 5... (period-2, every other line)
   → inactive lines carry global mesh h, no local control

6. apply_beta_transition(pi_nodes, e_nodes, sequence=(2,2,1))
   → sequence: 2 layers Pi-weighted, 2 layers mixed, 1 layer Β
   → each layer defined by distance along load axis from St. Venant boundary
   → 2-2-1 applied symmetrically from both Π faces inward

7. corner_inverted_triangle(corner_df, interior_nodes)
   → for each corner: find 3 nearest interior nodes
   → flag as inverted triangle pattern
   → recess corner node by h_corner × 0.5 inward along bisector

8. chainsaw_edge_pattern(edge_df, interior_nodes, load_dir)
   → for each active edge: find interior nodes within SOI radius
   → alternate: even-index nodes interior-linked, odd-index nodes boundary-linked
   → tooth depth = h_edge × 0.5

9. compute_topological_entropy(face_r, edge_r, corner_r, interior_r)
   → entity counts: n_Pi, n_E, n_B (surface + interior separately)
   → Β weight from 2-2-1: [2, 2, 1] / 5 = [0.4, 0.4, 0.2]
   → entropy = -Σ p_i × log(p_i) over weighted entity distribution
   → one scalar output

10. print_ansys_interior_instructions(all_results)
    → Body Sizing for Ε cylinder lines
    → Named Selection + Sizing for Pi center zone
    → Inflation layers for 2-2-1 Β transition
    → Corner and edge patterns as geometric corrections