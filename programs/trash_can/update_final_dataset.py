import Dataset
# Open the input file in read mode and output file in write mode
with open('dataset/final_dataset.txt', 'r') as infile, open('dataset/final_dataset_updated.txt', 'w') as outfile:
    # Iterate over each line in the input file
    for line in infile:
        # Strip the newline character from the end of the line
        line = line.rstrip('\n')
        # Append ',0' ten times to the line
        record = Dataset.DatasetRecord(line)
        if record.type == '1e':
            outfile.write(line + '\n')
            continue
        if record.type == 'j2_wo':
            record.e, record.d, record.c, record.b = record.d, record.c, record.b, record.a
            record.a = '0'
        # for _ in range(8):
        #     line += ',0'
     # Write the modified line to the output file
        outfile.write(record.to_text())