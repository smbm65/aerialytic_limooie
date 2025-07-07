# visualize_knn_segments_from_dataset.py
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import argparse
import os
import sys

def segment_distance(p1, q1, p2, q2, eps=1e-9):
    u = q1 - p1
    v = q2 - p2
    w0 = p1 - p2
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    denom = a * c - b * b
    t_u_inf, t_v_inf = 0.0, 0.0
    if denom > eps:
        t_u_inf = (b * e - c * d) / denom
        t_v_inf = (a * e - b * d) / denom
    else:
        if a < eps:
            t_u_inf = 0.0
            if c < eps: t_v_inf = 0.0
            else: t_v_inf = np.clip(-e / (c + eps), 0.0, 1.0)
        elif c < eps:
            t_v_inf = 0.0
            t_u_inf = np.clip(d / (a + eps), 0.0, 1.0)
        else:
            t_u_inf = 0.0
            t_v_inf = np.clip(-e / (c+eps), 0.0, 1.0)
    tc = np.clip(t_u_inf, 0.0, 1.0)
    sc = np.clip(t_v_inf, 0.0, 1.0)
    closest_point_on_segment1 = p1 + tc * u
    closest_point_on_segment2 = p2 + sc * v
    distance_vector = closest_point_on_segment1 - closest_point_on_segment2
    distance = np.linalg.norm(distance_vector)
    return distance


def get_text_offset(ax, x, y, existing_offsets, base_offset_dist=0.05, angle_step=np.pi/4, search_radius_factor=0.8):
    for i in range(8):
        angle = i * angle_step
        current_offset_dist = base_offset_dist * (1 + (i // 2) * 0.2)

        offset_x = current_offset_dist * np.cos(angle)
        offset_y = current_offset_dist * np.sin(angle)
        
        candidate_pos = (x + offset_x, y + offset_y)
        
        too_close = False
        for ex_x, ex_y in existing_offsets:
            if np.sqrt((candidate_pos[0] - ex_x)**2 + (candidate_pos[1] - ex_y)**2) < base_offset_dist * search_radius_factor :
                too_close = True
                break
        if not too_close:
            return offset_x, offset_y
            
    return base_offset_dist * 1.5 * np.cos(0), base_offset_dist * 1.5 * np.sin(0)


def are_segments_geometrically_identical(p1_a, q1_a, p1_b, q1_b, tol=1e-7):
    cond1 = np.allclose(p1_a, p1_b, atol=tol) and np.allclose(q1_a, q1_b, atol=tol)
    cond2 = np.allclose(p1_a, q1_b, atol=tol) and np.allclose(q1_a, p1_b, atol=tol)
    return cond1 or cond2


def main():
    parser = argparse.ArgumentParser(description="Visualize K-Nearest Segment Distances for a line in a graph from dataset JSON.")
    parser.add_argument('--viz_json_path', type=str, required=True, help='Path to the visualization/data JSON file.')
    parser.add_argument('--viz_graph_ids', type=str, nargs='+', required=True, help='One or more Graph IDs to visualize.')
    parser.add_argument('--target_line_index_in_graph', type=int, default=0,
                        help='0-based index of the line within the graph to use as the target for KNN visualization.')
    parser.add_argument('--k_to_show', type=int, default=5, help='Number of K nearest neighbors to annotate ON THE PLOT.')
    parser.add_argument('--output_dir', type=str, default='knn_segment_plots', help='Directory to save output plots.')
    parser.add_argument('--text_offset_scale', type=float, default=0.05, help="Scale factor for KNN text annotation offset from midpoint.")
    parser.add_argument('--show_all_indices', action='store_true', help="Show original index on all lines.")
    parser.add_argument('--index_text_offset_scale', type=float, default=0.03, help="Scale factor for original index text offset.")

    args = parser.parse_args()

    try:
        with open(args.viz_json_path, 'r', encoding='utf-8') as f:
            all_graph_data = json.load(f)
        print(f"Loaded data for {len(all_graph_data)} graphs from {args.viz_json_path}")
    except FileNotFoundError:
        print(f"FATAL ERROR: Visualization JSON file not found at '{args.viz_json_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"FATAL ERROR: Could not decode JSON from '{args.viz_json_path}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)


    for graph_id_to_viz in args.viz_graph_ids:
        print(f"\nProcessing Graph ID: {graph_id_to_viz}")
        graph_edges_data_from_json = all_graph_data.get(str(graph_id_to_viz))
        if not graph_edges_data_from_json:
            print(f"Warning: Graph ID '{graph_id_to_viz}' not found. Skipping.")
            continue
        if not isinstance(graph_edges_data_from_json, list) or not graph_edges_data_from_json:
            print(f"Warning: Data for Graph ID '{graph_id_to_viz}' is empty/invalid. Skipping.")
            continue

        current_graph_lines = []
        for i, edge_entry in enumerate(graph_edges_data_from_json):
            try:
                p1 = np.array([edge_entry['x1_norm'], edge_entry['y1_norm']])
                q1 = np.array([edge_entry['x2_norm'], edge_entry['y2_norm']])
                if np.linalg.norm(q1 - p1) < 1e-7:
                    continue 
                current_graph_lines.append({'id': f"L{i}", 'original_index': i, 'p1': p1, 'q1': q1, 'data': edge_entry})
            except KeyError as e:
                print(f"Warning: Missing key {e} for edge {i} in graph {graph_id_to_viz}. Skipping edge.")
            except Exception as e:
                print(f"Warning: Error processing edge {i} in graph {graph_id_to_viz}: {e}. Skipping edge.")
        
        if not current_graph_lines:
            print(f"Warning: No valid lines for graph {graph_id_to_viz}. Skipping.")
            continue

        num_edges_in_graph = len(current_graph_lines)
        print(f"Graph '{graph_id_to_viz}' has {num_edges_in_graph} processable lines.")

        target_line_info = None
        original_target_idx_search = args.target_line_index_in_graph
        for line_info_iter in current_graph_lines:
            if line_info_iter['original_index'] == original_target_idx_search:
                target_line_info = line_info_iter
                break
        
        if not target_line_info:
            print(f"Warning: Target line (orig. index {original_target_idx_search}) not found after filtering. Skipping graph.")
            continue

        p1_target = target_line_info['p1']
        q1_target = target_line_info['q1']

        distances_to_target = []
        for other_line_info in current_graph_lines:
            if other_line_info['original_index'] == target_line_info['original_index']:
                continue
            p2_other = other_line_info['p1']
            q2_other = other_line_info['q1']
            dist = segment_distance(p1_target, q1_target, p2_other, q2_other)
            is_identical = are_segments_geometrically_identical(p1_target, q1_target, p2_other, q2_other)
            distances_to_target.append({
                'original_index': other_line_info['original_index'],
                'distance': dist,
                'is_identical': is_identical,
                'data': other_line_info
            })

        distances_to_target.sort(key=lambda x: (not x['is_identical'], x['distance']))

        fig, ax = plt.subplots(figsize=(12, 10))
        plotted_text_positions = []

        for line_info in current_graph_lines:
            p, q = line_info['p1'], line_info['q1']
            ax.plot([p[0], q[0]], [p[1], q[1]],
                    color='skyblue', linewidth=1.5, marker='.', markersize=3, zorder=1)

            if args.show_all_indices:
                mid_x = (p[0] + q[0]) / 2
                mid_y = (p[1] + q[1]) / 2
                idx_offset_x, idx_offset_y = get_text_offset(ax, mid_x, mid_y, plotted_text_positions,
                                                               base_offset_dist=args.index_text_offset_scale,
                                                               angle_step=np.pi/2,
                                                               search_radius_factor=0.6)

                ax.text(mid_x + idx_offset_x, mid_y + idx_offset_y,
                        f"{line_info['original_index']}",
                        color='dimgray', fontsize=7, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.5, pad=0.1, boxstyle='round,pad=0.1'), zorder=2)
                plotted_text_positions.append((mid_x + idx_offset_x, mid_y + idx_offset_y))

        ax.plot([p1_target[0], q1_target[0]], [p1_target[1], q1_target[1]],
                color='red', linewidth=3, marker='o', markersize=5, zorder=3,
                label=f"Target Line (Orig. Index: {target_line_info['original_index']})")

        print(f"\nDistances from Line Orig. Index {target_line_info['original_index']} in graph '{graph_id_to_viz}':")
        for rank, neighbor_info in enumerate(distances_to_target):
            identity_marker = "*" if neighbor_info['is_identical'] else ""
            print(f"  Rank {rank+1}{identity_marker}: Line Orig. Index {neighbor_info['original_index']}, Distance: {neighbor_info['distance']:.3f}")

        print(f"\nAnnotating top {args.k_to_show} neighbors on the plot:")
        for rank, neighbor_info in enumerate(distances_to_target[:args.k_to_show]):
            neighbor_p1 = neighbor_info['data']['p1']
            neighbor_q1 = neighbor_info['data']['q1']

            neighbor_color = 'purple' if neighbor_info['is_identical'] else 'darkorange'
            neighbor_linestyle = '-' if neighbor_info['is_identical'] else '--'
            neighbor_linewidth = 2.5 if neighbor_info['is_identical'] else 2.0

            ax.plot([neighbor_p1[0], neighbor_q1[0]], [neighbor_p1[1], neighbor_q1[1]],
                    color=neighbor_color, linewidth=neighbor_linewidth, linestyle=neighbor_linestyle, zorder=2)

            mid_x = (neighbor_p1[0] + neighbor_q1[0]) / 2
            mid_y = (neighbor_p1[1] + neighbor_q1[1]) / 2
            
            knn_offset_x, knn_offset_y = get_text_offset(ax, mid_x, mid_y, plotted_text_positions,
                                                           base_offset_dist=args.text_offset_scale)
            
            label_text = f"K{rank + 1}"
            if neighbor_info['is_identical']:
                label_text += "*"

            ax.text(mid_x + knn_offset_x, mid_y + knn_offset_y, label_text, color='black', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, pad=0.2, boxstyle='round,pad=0.3'), zorder=4,
                    ha='center', va='center')
            plotted_text_positions.append((mid_x + knn_offset_x, mid_y + knn_offset_y))
            

        ax.set_title(f"KNN (Segment Dist) to Line {target_line_info['original_index']} in Graph '{graph_id_to_viz}'")
        ax.set_xlabel("X-coordinate (Normalized)")
        ax.set_ylabel("Y-coordinate (Normalized)")
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()

        output_filename = os.path.join(args.output_dir, f"graph_{graph_id_to_viz}_target_{target_line_info['original_index']}_knn.png")
        plt.savefig(output_filename)
        print(f"Saved KNN visualization to {output_filename}")
        plt.close(fig)

    print("\nVisualization script finished")

if __name__ == "__main__":
    main()
    