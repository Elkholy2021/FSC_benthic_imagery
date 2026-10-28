import numpy as np


def imbalance_ratio(counts):
    counts = np.array(counts)
    return counts.max() / counts.min()


def coefficient_variation(counts):
    counts = np.array(counts)
    return np.std(counts) / np.mean(counts)


def entropy_imbalance(counts):
    counts = np.array(counts)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    Hmax = np.log(len(counts))
    return 1 - H / Hmax

def gini_coefficient(counts):
    counts = np.array(counts)
    diff_sum = np.sum(np.abs(counts[:, None] - counts))
    return diff_sum / (2 * len(counts) * counts.sum())




dataset_A = {"Ball_Yellow_Papillate_Irregular": 534, 
             "Cup_Beige_Thick": 125, 
             "Cup_Black_Smooth": 109, 
             "Cup_Orange": 101,
            "Cup_Red_Smooth": 887,
            "Cup_Red_Thick": 606,
            "Cup_Yellow": 560,
            "Fan_Pink": 237,
            "Massive_Purple": 429}
            

dataset_B = {"Rock": 1946, 
             "Red": 888, 
             "Screwshells": 547, 
             "Brittle_snake_stars": 232}



dataset_C = {"Pebble_gravel": 2293, 
             "Turf_Sediment_Silt_matrix": 2197, 
             "Sand_mud_2mm": 2181, 
             "Bryozoa_Cnidaria_Matrix": 2161,
             "Biologenic":886,
             "Antler":173}
if __name__ == "__main__":
    datasets = (dataset_A,dataset_B, dataset_C)
    for dataset in datasets:
        counts = list(dataset.values())
        dataset_name = next((name for name, obj in globals().items() if obj is dataset), None)
        print("Dataset:", dataset_name)
        print("Counts:", counts)
        print(f"Imbalance Ratio: {imbalance_ratio(counts):.4f}")
        print(f"Coefficient of Variation: {coefficient_variation(counts):.4f}")
        print(f"Entropy Imbalance: {entropy_imbalance(counts):.4f}")
        print(f"Gini Coefficient: {gini_coefficient(counts):.4f}")
        print(f"max class count: {max(counts)}")
        print(f"min class count: {min(counts)}")
        print(f"total count: {sum(counts)}")
        print("\n")
        print("====================================\n")
