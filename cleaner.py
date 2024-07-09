def remove_blank_lines(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.strip():
                outfile.write(line)

if __name__ == "__main__":
    input_file = 'my-clean-games.pgn'  # Replace with your input file name
    output_file = 'kush-clean-games.pgn'  # Replace with your desired output file name
    remove_blank_lines(input_file, output_file)