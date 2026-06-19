import argparse
import sys
import subprocess

def main(dataset, model, task, export_dataset):
    print("Running with the following parameters:")
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"Export Dataset: {export_dataset}")

    if task == "enc":
        args = [
            'python', 'newclass.py',
            '--setting', dataset,
            '--model', model,
            '--export_dataset', str(export_dataset).lower()
        ]

        # 调用 script.py
        result = subprocess.run(args, capture_output=True, text=True)

        # 打印输出
        print('STDOUT:', result.stdout)
        print('STDERR:', result.stderr)
    elif task == "de":
        args = [
            'python', 'run_experiment.py',
            '--dataset', dataset,
            '--model', model,
            '--task', "random",
            '--degree', "all",
            '--export_dataset', str(export_dataset).lower()
        ]

        result = subprocess.run(args, capture_output=True, text=True)

        print('STDOUT:', result.stdout)
        print('STDERR:', result.stderr)
    elif task == "ds" or task == "vb":
        args = [
            'python', 'distribution_objectives.py',
            '--setting', dataset,
            '--model', model,
            '--export_dataset', str(export_dataset).lower()
        ]

        result = subprocess.run(args, capture_output=True, text=True)

        print('STDOUT:', result.stdout)
        print('STDERR:', result.stderr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a machine learning task with specified parameters.")

    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use.')
    parser.add_argument('--model', type=str, required=True, help='Model to use for the task.')
    parser.add_argument('--task', type=str, required=True, help='Task to perform (e.g., classification, regression).')
    parser.add_argument('--export_dataset', type=lambda x: (str(x).lower() == 'true'), default=False, required=False, help='Whether to export the dataset after processing.')

    args = parser.parse_args()

    main(args.dataset, args.model, args.task, args.export_dataset)