import csv
import torch

def save_results_to_csv(temp_output_file, output_csv):
    """
    Reads inference results from a temporary file and saves them to a CSV.

    Args:
        temp_output_file (str): Path to the temporary file with results.
        output_csv (str): Path to save the prediction results.
    """
    # Load intermediate results
    results = torch.load(temp_output_file)

    # Save results to CSV
    with open(output_csv, mode='w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "PredictedLabel"])  # Header
        writer.writerows(results)

    print(f"Inference results saved to CSV file '{output_csv}'.")

if __name__ == "__main__":
    temp_output_file = "results.pt"  # Temporary file
    output_csv = "test_predictions.csv"
    save_results_to_csv(temp_output_file, output_csv)
