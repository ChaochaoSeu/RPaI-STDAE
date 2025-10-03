import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import os
import pickle


def create_interchange_adjacency_matrix(df_turn, interchange_name):
    """
    Create interchange adjacency matrix

    Parameters:
    df_turn: Interchange basic information DataFrame
    interchange_name: Name of the interchange

    Returns:
    adjacency_matrix: Adjacency matrix
    distance_matrix: Distance matrix
    """

    # Filter data for the specified interchange
    filtered_data = df_turn[df_turn['INTERCHANGE'] == interchange_name].copy()

    # Reset index
    filtered_data = filtered_data.reset_index(drop=True)

    # Turn direction order: East->North, East->West, East->South, West->South, West->East, West->North,
    # South->East, South->North, South->West, North->West, North->South, North->East
    turndirection_order = [
        "东向北", "东向西", "东向南", "西向南", "西向东", "西向北",
        "南向东", "南向北", "南向西", "北向西", "北向南", "北向东"
    ]

    # Get number of turns
    n_turn = len(turndirection_order)

    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((n_turn, n_turn), dtype=float)

    # Calculate distance matrix
    distance_matrix = np.zeros((n_turn, n_turn))

    for i, dir_i in enumerate(turndirection_order):
        for j, dir_j in enumerate(turndirection_order):
            if i != j:
                # Connect if they share the same origin direction (first character) or destination direction (third character)
                if (dir_i[0] == dir_j[0]) or (dir_i[2] == dir_j[2]):
                    adjacency_matrix[i, j] = 1.0
                    distance_matrix[i, j] = 100.0

    return adjacency_matrix, distance_matrix

def generate_adj_QiLin():
    # File path
    file_path = r""  # Gantry basic information file path
    # Read specified sheet
    df_turn = pd.read_excel(file_path, sheet_name="gantryinfo_turn")

    # Create adjacency matrix
    adj_mx, distance_mx = create_interchange_adjacency_matrix(
        df_turn,
        interchange_name="麒麟枢纽")  # QiLin Interchange

    # The self loop is missing
    add_self_loop = False

    if add_self_loop:
        print("Adding self loop to adjacency matrices.")
        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
        distance_mx = distance_mx + np.identity(distance_mx.shape[0])
    else:
        print("Kindly note that there is no self loop in adjacency matrices.")

    # Save adjacency matrices
    with open("datasets/InterchangeData/QiLin/adj_QiLin.pkl", "wb") as f:
        pickle.dump(adj_mx, f)
    with open("datasets/InterchangeData/QiLin/adj_QiLin_distance.pkl", "wb") as f:
        pickle.dump(distance_mx, f)