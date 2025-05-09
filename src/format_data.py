import os
import shutil
import pwd
glob_imported = False

try:
    import glob
    glob_imported = True
except ImportError:
    print("glob module is required. Please install it using 'pip install glob2'.")


def organize_assets(input_folder):
    if not glob_imported:
        print("Cannot proceed without the 'glob' module.")
        return
    
    # Create the main data directory
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Process each CSV file in the input directory
    csv_files = glob.glob(os.path.join(input_folder, "stocks - *.csv"))
    for file_path in csv_files:
        # Extract the asset name from the file name
        file_name = os.path.basename(file_path)
        asset_name = file_name.split('_')[-1].split('.')[0]
        
        # Create the asset directory
        asset_dir = os.path.join(data_dir, asset_name)
        os.makedirs(asset_dir, exist_ok=True)
        
        # Copy the file with the standardized name
        dest_file = os.path.join(asset_dir, 'simulation_steps.csv')
        shutil.copy(file_path, dest_file)
        print(f"Processed: {file_name} -> {dest_file}")


if __name__ == '__main__':
    # input_folder = input("Enter the path to the folder containing your CSV files: ").strip()
    input_folder = 'google_data'
    organize_assets(input_folder)
