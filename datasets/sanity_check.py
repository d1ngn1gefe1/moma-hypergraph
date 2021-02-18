from datasets.moma_api import get_momaapi
import numpy as np
from pprint import pprint


"""
Inconsistent bounding box: 20201201142505

Object cid is null: 20201115195620, 20201110150303

Wrong instance id:
  20201103132536 (8): {'A ': 8, 'B': 8}, {'1': 8, '2': 8, '3': 6, '4': 8, '5': 8, '99': 8}, {'A': 24, 'B': 22}
  20201106213401 (9): {'A ': 5}, {'1': 9, '2': 5}, {'A': 7, 'A ': 7}
  20201106222226 (6): {'2': 3, 'A ': 6}, {'1': 6, '2': 6}, {'A': 1, 'B': 3}
  20201110150625 (8): {'A': 8}, {'1': 8, '3': 8, 'unlabeled': 8}, {'A': 16}
  20201110160043 (10): {'A ': 10, 'B ': 10}, {'1': 9}, {'A': 10, 'A ': 4}
  20201112143946 (5): {'A': 5, 'B': 3}, {'unlabeled': 2}, {'A': 5, 'B': 3}
  20201113101742 (4): {'A': 4, 'B': 4, 'c': 3}, {'1': 4, '2': 3}, {'A': 6, 'B': 6, 'C': 3}
  20201114004750 (13): {'A ': 13, 'B': 13, 'C': 13}, {'1': 13, '2': 13, '3': 13, '4': 1, '5': 12}, {'A': 39, 'B': 13, 'C': 22}
  20201114200234 (30): {'A': 30, 'B ': 30}, {'1': 8, '2': 14, '3': 15, '4': 41}, {'(1)': 4, '(2)': 6, '(A)': 6, '(B)': 4, '1': 6, 'A': 90, 'B': 33}
  20201114205126 (6): {'A': 6, 'B': 6, 'C ': 6}, {'1': 6, '2': 6, '3': 6, '4': 6, '5': 6, '6': 6}, {'A': 18, 'A ': 3, 'B': 12, 'C': 6}
  20201115020327 (6): {'A ': 6, 'B ': 6, 'C': 6, 'D': 6, 'E': 6, 'F': 6, 'G': 6, 'H': 6, 'I': 6, 'J': 6, 'K': 6, 'L': 6, 'M': 6, 'N': 4, 'O': 6}, {'1': 6}, {'A': 12, 'B': 14, 'C': 12, 'D': 12, 'E': 12, 'F': 12, 'G': 12, 'H': 12, 'I': 12, 'J': 12, 'K': 12, 'L': 12, 'M': 12, 'N': 9, 'O': 12}
  20201115174604 (31): {'A ': 31, 'B': 31}, {'1': 31}, {'A': 31, 'B': 31}
  20201115202013 (26): {'A': 11, 'A ': 13, 'B': 26}, {'1': 19, '2': 15}, {'A': 24}
  20201115222454 (60): {'A ': 60, 'B ': 60}, {'1': 60}, {'A': 58, 'B': 60}
  20201116180159 (7): {'A': 5, 'A ': 1, 'B': 7, 'C': 2, 'D': 2, 'F': 1}, {'1': 2, '1 ': 5, '2': 7, '3': 2, '4': 2, '7': 2, '8': 2, '9': 2}, {'A': 14, 'B': 14, 'C': 4, 'D': 4}
  20201116185344 (6): {'A': 7, 'B': 5, 'C': 6}, {'1': 5, '2': 2, '3': 2, '4': 1, 'unlabeled': 2}, {'B': 14, 'C': 6}
  20201117105120 (4): {'1': 4}, {'1': 4, '2': 4}, {'A': 8}
  20201117105213 (23): {'A': 17, 'A      ': 3, 'B': 2}, {'1': 8, '2': 1}, {'A': 22}
  20201117132415 (7): {'A': 6, 'B ': 6}, {'1': 5}, {'A': 6, 'B': 7}
  20201123175945 (3): {'B': 3, 'C': 3, 'D': 3, 'E': 3, 'F': 3, 'G': 1}, {'1': 3, 'A': 3}, {'A': 6, 'B': 3, 'C': 9, 'D': 3, 'E': 3, 'F': 3, 'G': 1}
  20201130201632 (5): {'A ': 5, 'B ': 5}, {'1': 10, '2': 5}, {'A': 10, 'B': 5}
  20201201000457 (23): {'A': 6, 'B': 8, 'C': 6, 'D': 6, 'E': 3, 'F': 7, 'G': 6, 'H': 4, 'I': 5, 'J': 5, 'K': 5, 'L': 6, 'M': 6, 'N': 5, 'O': 6, 'P': 6, 'Q': 8, 'R': 6, 'S': 3, 'T': 7, 'U': 6, 'V': 6, 'W': 4, 'X': 7, 'Y': 6, 'Z': 2}, {'1': 16, 'B': 3}, {'(B': 2, 'A': 10, 'B': 24, 'C': 18, 'D': 18, 'E': 8, 'F': 11, 'G': 6, 'G)': 2, 'H': 4, 'I': 6, 'J': 6, 'K': 5, 'L': 6, 'M': 6, 'N': 5, 'O': 7, 'P': 6, 'Q': 6, 'Q.R': 1, 'R': 5, 'S': 2, 'T': 7, 'U': 6, 'V': 6, 'W': 4, 'X': 3, 'Y': 7, 'Z': 2}
  20201201013806 (60): {'A': 4, 'A ': 55, 'B': 60}, {'1': 55}, {'A': 107}
  20201201142349 (5): {'A ': 5, 'B ': 5, 'C': 5}, {'1': 5, '2': 5, '3': 5, '4': 5, '5': 5}, {'A': 10, 'B': 10, 'C': 10}
  20201201142408 (9): {'A ': 9, 'B ': 9, 'C ': 9}, {'1': 9, '2': 5, '3': 4, '4': 9, '5': 9, '6': 9, '7': 9, '8': 6, '9': 9}, {'A': 26, 'A ': 9, 'B': 36, 'C': 19}
  20201201142706 (6): {'A': 1, 'B': 6, 'C': 4, 'D': 5, 'c': 1}, {'1': 6, '2': 5}, {'B': 12, 'C': 19, 'D': 8}
  20201201143140 (27): {'A': 27, 'B': 27}, {'(4)': 4, '1': 10, '2': 60, '3': 3, '4': 8}, {'A': 27}
  20201201152651 (14): {'A ': 5}, {'1': 14}, {'A': 3, 'A ': 1}
  20201203211234 (62): {'A': 60, 'B': 32, 'C': 3}, {'(A),(5)': 2, '1': 21, '2': 4, '3': 3, '4': 11, '6': 14, '7': 6}, {'A': 104, 'B': 15, 'C': 2}
  20201204091616 (6): {'A': 6, 'B ': 6}, {'1': 6, '2': 6, '3': 5, '4': 5}, {'A': 17, 'B': 17}
  20201206004857 (38): {'A': 1, 'B': 25, 'F': 1, 'Ｃ': 32, 'Ｄ': 32}, {'1': 1, '2': 22, '3': 22, '4': 4, '5': 1, '6': 1, '7': 18, '8': 6, '9': 1}, {'A': 1, 'B': 26, 'C': 43, 'D': 32, 'E': 16, 'Ｃ': 14, 'Ｄ': 22}
  20201207144808 (11): {'A': 7, 'B': 11}, {'1': 5, '2': 11, '3\n': 10, '4': 7}, {'A': 18, 'B': 11}
  20201207214811 (7): {'2': 7, 'A': 7, 'C': 7, 'D': 7, 'E': 7, 'F': 7, 'G': 7}, {'1': 6}, {'A': 12, 'B': 14, 'C': 14, 'D': 14, 'E': 14, 'F': 7, 'G': 7}
  20201207220450 (5): {'2': 5, 'A': 5}, {'1': 5, '2': 5, '3': 5, '4': 5}, {'A': 10, 'B': 10}
  20201207222711 (19): {'2': 13, '3': 19, 'A': 19}, {'1': 19, '2': 12, '3': 9, '4': 19}, {'A': 70, 'B': 34, 'C': 19}
  20201207224252 (8): {'2': 8, 'A': 8, 'C': 8}, {'1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 6}, {'A': 31, 'B': 14, 'C': 8}
  20201207224512 (12): {'11': 3, 'A': 12, 'B': 12, 'C': 12, 'D': 12, 'E': 12, 'F': 12, 'G': 12, 'H': 12, 'I': 9, 'J': 11, 'K': 8, 'L': 5}, {'1': 12, '10': 10, '11': 9, '12': 10, '2': 12, '3': 12, '4': 10, '5': 12, '6': 4, '7': 12, '8': 7, '9': 10}, {'A': 35, 'B': 24, 'C': 12, 'D': 12, 'E': 21, 'F': 12, 'G': 19, 'H': 12, 'J': 12, 'K': 12}
  20201208023518 (7): {'A': 6, 'B': 7, 'C': 7, 'D': 7, 'E': 7, 'F': 7, 'G': 7, 'H': 7, 'I': 7, 'J': 7, 'M': 4, 'N': 5, 'O': 7, 'P': 5, 'Q': 2, 'k': 7, 'l': 7}, {'1': 7}, {'A': 13, 'B': 14, 'C': 14, 'D': 14, 'E': 14, 'F': 14, 'G': 14, 'H': 14, 'I': 14, 'J': 14, 'K': 14, 'L': 14, 'M': 11, 'N': 13, 'O': 14, 'P': 13, 'Q': 7}
  20201214173720 (45): {'A': 45, 'B': 45}, {'1': 20, '2': 38, '3': 18, '3                ': 26, '4': 16, '5': 8, '6': 24, '7': 24}, {'A': 65}

Bad relationship:
  20201102205649: [['A', ')(1', '2']]
  20201102205649: [['A', ')(1', '2']]
  20201102205649: [['A', ')(1', '2']]
  20201102205649: [['A', ')(1', '2']]
  20201102205649: [['A', ')(1', '2']]
  20201102205720: [['A)在……一旁(B']]
  20201102205720: [['A)在……一旁(B']]
  20201102205720: [['A)在……一旁(B']]
  20201106211404: [['']]
  20201110124258: [['B', 'C', 'D', 'E']]
  20201110124258: [['B', 'E', 'F']]
  20201110124258: [['B', 'E', 'F']]
  20201110124258: [['']]
  20201111010117: [['A)(1', '2']]
  20201111010203: [['A', '1', '2']]
  20201111010203: [['A', '1', '2']]
  20201111010203: [['A', '1', '2']]
  20201111010203: [['A', '1', '2']]
  20201111013740: [['E', 'F', 'B']]
  20201112144107: [['A', '1']]
  20201112144107: [['A', '1']]
  20201112144107: [['A', '1']]
  20201112144107: [['B.2']]
  20201112144451: [['A', 'C']]
  20201112144451: [['A', 'C']]
  20201112144451: [['A', 'C']]
  20201112144451: [['A', 'C']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201112230847: [['A', 'B']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201114204432: [['A', 'B)(C']]
  20201115012504: [['']]
  20201115012530: [['A', 'B', 'C', 'D', 'E', 'F)(1']]
  20201115012530: [['E', 'F)(1']]
  20201115020712: [['B', '1']]
  20201115020712: [['B', '1']]
  20201115020712: [['B', '1']]
  20201115020712: [['B', '1']]
  20201115020712: [['B', '1']]
  20201115020712: [['B', '1']]
  20201115020954: [['B', '1)(A']]
  20201115020954: [['B', '1)(A']]
  20201115020954: [['B', '1)(A']]
  20201115020954: [['B', '1)(A']]
  20201115020954: [['B', '1)(A']]
  20201115020954: [['B', '1)(A']]
  20201115020954: [['B', '1)(A']]
  20201115191705: [['A', 'B)(3', '2']]
  20201115191705: [['A', 'B)(3', '2']]
  20201115191705: [['A', 'B)(3', '2']]
  20201115191705: [['A', 'B)(3', '2']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115192148: [['']]
  20201115195445: [['C', 'A']]
  20201115195445: [['C', 'A']]
  20201115195445: [['C', 'A']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115200628: [['B)', '']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115201311: [['A', 'B']]
  20201115221809: [['A)(B']]
  20201115235404: [['A', 'B', 'D)(C', 'E', 'F']]
  20201115235404: [['A', 'B', 'D)(C', 'E', 'F']]
  20201115235404: [['A', 'B', 'D)(C', 'E', 'F']]
  20201115235404: [['A', 'B', 'D)(C', 'E', 'F']]
  20201115235404: [['A', 'B', 'D)(C', 'E', 'F']]
  20201115235404: [['A', 'B', 'D)(C', 'E', 'F']]
  20201115235404: [['A', 'B', 'D)(C', 'E', 'F']]
  20201115235404: [['A', 'B', 'D)(C', 'E', 'F']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116143620: [['B', 'A']]
  20201116143620: [['B)(A']]
  20201116143620: [['B)(1']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175022: [['']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175614: [['A', '1']]
  20201116175824: [['A', '3']]
  20201116175824: [['A', '3']]
  20201116175824: [['A', '3']]
  20201116175824: [['A', '3']]
  20201116175824: [['A', '3']]
  20201116175824: [['A', '3']]
  20201116175824: [['A', '3']]
  20201116183253: [['A', '(1']]
  20201116183253: [['A', '(1']]
  20201116185243: [['A', 'B)(1']]
  20201116185243: [['A', 'B)(1']]
  20201116185243: [['A', 'B)(1']]
  20201117130941: [['B', 'C', 'D']]
  20201117130941: [['A', 'B', 'C', 'D']]
  20201117130941: [['A', 'B', 'C', 'D']]
  20201117130941: [['A', 'B', 'C', 'D']]
  20201117130941: [['A', 'B', 'C', 'D']]
  20201117130941: [['A', 'B', 'C', 'D']]
  20201117130941: [['A', 'B', 'C', 'D']]
  20201117130941: [['A', 'B', 'C', 'D']]
  20201117155633: [['']]
  20201117155633: [['']]
  20201117155633: [['']]
  20201117155633: [['']]
  20201117155633: [['']]
  20201123143648: [['A', 'B']]
  20201123143648: [['A', 'B']]
  20201123143648: [['A', 'B']]
  20201123143648: [['A', 'B']]
  20201123143648: [['A', 'B']]
  20201123143648: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123151731: [['A', 'B']]
  20201123180114: [['C)', ' (1']]
  20201123180114: [['C)', ' (1']]
  20201124141012: [['A', 'B', 'C)(1']]
  20201124141012: [['A', 'B', 'C)(1']]
  20201130175629: [['A', 'B)(1']]
  20201130175629: [['A', 'B)(1']]
  20201130175629: [['A', 'B)(1']]
  20201130175639: [['A', 'B)(1']]
  20201130175639: [['A', 'B)(1']]
  20201130175639: [['A', 'B)(1']]
  20201130175639: [['A', 'B)(1']]
  20201130175639: [['A', 'B)(1']]
  20201130175717: [['A', 'B)(4']]
  20201130175717: [['A', 'B)(4']]
  20201130175717: [['A', 'B)(4']]
  20201130202016: [['']]
  20201130202016: [['']]
  20201130202016: [['']]
  20201130202016: [['']]
  20201130202016: [['']]
  20201130202016: [['']]
  20201130202016: [['']]
  20201130202612: [['']]
  20201201000637: [['A', 'B']]
  20201201000637: [['A', 'B']]
  20201201131917: [['C)(A']]
  20201201133410: [['B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201143906: [['A', 'B']]
  20201201151022: [['A).(1']]
  20201201151022: [['A).(1']]
  20201201151022: [['A).(1']]
  20201201151022: [['A).(1']]
  20201201151022: [['A).(1']]
  20201201153542: [['A', 'C', 'B']]
  20201203203923: [['A', 'B)(1', '2']]
  20201203203923: [['A', 'B)(1', '2']]
  20201203203923: [['A', 'B)(1', '2']]
  20201203203923: [['A', 'B)(1', '2']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201203210825: [['']]
  20201204091941: [['']]
  20201204091941: [['']]
  20201204091941: [['']]
  20201204091941: [['']]
  20201204091941: [['']]
  20201204091941: [['']]
  20201204091941: [['']]
  20201204091941: [['']]
  20201207142255: [['A', 'B', 'C']]
  20201207142255: [['A', 'B', 'C']]
  20201207142255: [['A', 'B', 'C']]
  20201207142255: [['A', 'B', 'C']]
  20201207142255: [['A', 'B', 'C']]
  20201207153654: [['', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '']]
  20201207153654: [['', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '']]
  20201207153654: [['', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '']]
  20201207153654: [['', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '']]
  20201207153654: [['', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '']]
  20201207153751: [['C)', ' (D']]
  20201207153751: [['C)', ' (D']]
  20201207153751: [['C)', ' (D']]
  20201207154639: [['A', '1)(B']]
  20201207154639: [['A', '1)(B']]
  20201207154639: [['A', '1)(B']]
  20201207154639: [['A', '1)(B']]
  20201207154639: [['A', '1)(B']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L', 'M', 'N']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L', 'M', 'N']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L', 'M', 'N']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L', 'M', 'N']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L', 'M', 'N']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L', 'M', 'N']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L', 'M', 'N.P']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L', 'M', 'N.P']]
  20201207160725: [['A', 'B', 'C', 'D)(E', 'F', 'G', 'H', 'L', 'M', 'N', 'P', 'J']]
  20201207213926: [['A', 'B', 'C', 'D', 'E', 'M)(F', 'G', 'H', 'L']]
  20201207213926: [['A', 'B', 'C', 'D', 'E', 'M)(F', 'G', 'H', 'L']]
  20201207213926: [['A', 'B', 'C', 'D', 'E', 'M)(F', 'G', 'H', 'L']]
  20201207213926: [['A', 'B', 'C', 'D', 'E', 'M)(F', 'G', 'H', 'L']]
  20201207213926: [['A', 'B', 'C', 'D', 'E', 'M)(F', 'G', 'H', 'L', 'N']]
  20201207213926: [['A', 'B', 'C', 'D', 'E', 'M)(F', 'G', 'H', 'L', 'N']]
  20201207213926: [['A', 'B', 'C', 'D', 'E', 'M)(F', 'G', 'H', 'L', 'N']]
  20201207213926: [['A', 'B', 'C', 'D', 'E', 'M)(F', 'G', 'H', 'L', 'N']]
  20201207214117: [['']]
  20201207214130: [['', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '']]
  20201207214130: [['', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '', 'K', '']]
  20201207214130: [['', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '', 'K', '']]
  20201207214130: [['', 'C', 'D', 'E', 'F', 'G', 'H', '', 'K', '']]
  20201207214130: [['', 'C', 'D', 'E', 'F', 'G', 'H', '', 'K', '']]
  20201207214130: [['', 'C', 'D', 'E', 'F', 'G', 'H', '', 'K', '']]
  20201207214130: [['', 'C', 'D', 'E', 'F', 'G', 'H', '', 'K', '']]
  20201207214130: [['', 'C', 'D', 'E', 'F', 'G', 'H', '', 'K', '']]
  20201207215605: [['B', 'A']]
  20201207215605: [['B', 'A']]
  20201207215605: [['B', 'A']]
  20201207215605: [['B', 'A']]
  20201207215605: [['B', 'A']]
  20201207215839: [['A', 'B)(4']]
  20201207215839: [['A', 'B)(4']]
  20201207215839: [['A', 'B)(4']]
  20201207215839: [['A', 'B)(4']]
  20201207215839: [['A', 'B)(4']]
  20201207215839: [['A', 'B)(5']]
  20201207222132: [['A', 'B', 'C', 'D', 'E', '']]
  20201207222132: [['A', 'B', 'C', 'D', 'E', '']]
  20201207222132: [['A', 'B', 'C', 'D', 'E', '']]
  20201207222132: [['A', 'B', 'C', 'D', 'E', '']]
  20201207222132: [['A', 'B', 'C', 'D', 'E', '']]
  20201207222132: [['A', 'B', 'C', 'D', 'E', '']]
  20201207222132: [['A', 'B', 'C', 'D', 'E', '']]
  20201207223932: [['A', '(C']]
  20201207225434: [['A', 'B)(1']]
  20201207231050: [['O', 'N']]
  20201207231050: [['O', 'N']]
  20201207231050: [['O', 'N']]
  20201207231050: [['O', 'N']]
  20201207231050: [['O', 'N']]
  20201214180701: [['B', 'A']]
  20201214180701: [['B', 'A']]
  20201214180701: [['B', 'A']]
  20201214184415: [['']]
  20201214184415: [['D)(', 'F']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184415: [['']]
  20201214184945: [['A', 'E', 'F)', 'H', '(1', '2']]
"""


def main():
  dataset_dir = '/Users/alanzluo/Documents/moma'
  api = get_momaapi(dataset_dir, 'trimmed_video')
  trimmed_video_ids = sorted(api.annotations.keys())

  for trimmed_video_id in trimmed_video_ids:
    annotation = api.get_annotation(trimmed_video_id)
    graphs = annotation['graphs']

    actor_instance_ids, object_instance_ids, atomic_action_actor_instance_ids, relationship_instance_ids = [], [], [], []

    for graph in graphs:
      for actor in graph['actors']:
        actor_instance_ids.append(actor['instance_id'])

      for object in graph['objects']:
        object_instance_ids.append(object['instance_id'])

      for action in graph['atomic_actions']:
        atomic_action_actor_instance_ids += action['actor_instance_ids']

      for relationship in graph['relationships']:
        description = relationship['description']
        try:
          relationship_instance_ids += description[0]+description[1]
        except:
          print('{}: {}'.format(trimmed_video_id, description))

    # if all([len(x) == 1 and x.isalpha() and x.isupper() for x in np.unique(actor_instance_ids)]) and \
    #     all([x.isdigit() for x in np.unique(object_instance_ids)]):
    #   continue

    # actor_info = {i: j for i, j in zip(*np.unique(actor_instance_ids, return_counts=True))}
    # object_info = {i: j for i, j in zip(*np.unique(object_instance_ids, return_counts=True))}
    # atomic_action_info = {i: j for i, j in zip(*np.unique(atomic_action_actor_instance_ids, return_counts=True))}
    # relationship_info = {i: j for i, j in zip(*np.unique(relationship_instance_ids, return_counts=True))}
    # print('{} ({}): {}, {}, {}, {}'.format(trimmed_video_id, len(graphs), actor_info, object_info, atomic_action_info, relationship_info))


if __name__ == '__main__':
  main()
