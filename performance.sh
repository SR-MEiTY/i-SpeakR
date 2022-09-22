#!/bin/bash
# The groundtruth file must have the "utterance_id" and "true_utterance_id" fields
# The prediction file must have the "utterance_id", "speaker_id" and "score" fields
# The "utterance_id" field must contain the random utterance ids for the test files. 
# The "true_utterance_id" field must contain the original utterance id of the files
# The program will output the results in the terminal, as well as to a "Evaluation_Results.txt" file

# After the --groundtruth switch, provide the path to the groundtruth csv file.
# After the --prediction switch, provide the path to the prediction csv file
# The <> is provide just for representation purposes. It is not required while running the script

python evaluation.py --groundtruth <> --prediction <>

