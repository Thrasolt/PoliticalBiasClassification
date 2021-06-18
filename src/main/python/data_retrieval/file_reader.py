import csv
from typing import List, Tuple
left_path = "/home/thrasolt/git/home_project_students/input_data/left_train.csv"
right_path = "/home/thrasolt/git/home_project_students/input_data/right_train.csv"
test_path = "/home/thrasolt/git/home_project_students/input_data/test.csv"


def read_training_data(
        left_pld: str = left_path,
        right_pld: str = right_path) -> List[Tuple[str, int]]:

    with open(left_pld, newline='\n') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        left = [(row[0], 0) for row in reader]
    with open(right_pld, newline='\n') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        right = [(row[0], 1) for row in reader]
    return left+right


def read_plds(
        test_pld: str = test_path):
    with open(test_pld, newline='\n') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
        test_data = [(row[0], -1) for row in reader]
    return test_data


