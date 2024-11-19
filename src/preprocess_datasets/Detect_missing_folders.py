import os


def find_missing_folders(directory):
    # Get list of all folders
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Extract folder names as integers, assuming they are numerical
    folder_numbers = sorted([int(folder) for folder in folders if folder.isdigit()])

    # Find missing folder numbers
    missing_folders = [i for i in range(folder_numbers[0], folder_numbers[-1] + 1) if i not in folder_numbers]

    print("Missing folders:", len(missing_folders))

    # Format output with newlines every 'th' folder
    formatted_output = ''
    for index, folder in enumerate(missing_folders):
        formatted_output += str(folder)
        if (index + 1) % 25 == 0:  # Add newline after every 'th' folder
            formatted_output += '\n'
        else:
            formatted_output += ', '  # Optional: separate with commas

    return formatted_output.strip()  # Remove trailing whitespace/newline


directory = "H:\\Maurer\\FFHQ-MonoNPHM\\stage1\\"
missing_folders = find_missing_folders(directory)
print("Missing folders:", missing_folders)

