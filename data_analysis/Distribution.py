import argparse
import os
from pyfiglet import Figlet
import matplotlib.pyplot as plt
import numpy as np

f = Figlet(font='slant')

def check_args(args):
    try:
        if not os.path.exists(args.dataset):
            raise Exception(f'Path {args.dataset} does not exist')
        if not os.path.isdir(args.dataset):
            raise Exception(f'Path {args.dataset} is not a directory')
        if not os.listdir(args.dataset):
            raise Exception(f'Directory {args.dataset} is empty')
    except Exception as e:
        print(f'\033[91m{e}\033[0m')
        exit(1)

def get_distribution(dataset):
    classes = os.listdir(dataset)
    distribution = {}
    for c in classes:
        distribution[c] = len(os.listdir(os.path.join(dataset, c)))
    return distribution

def pie_chart(distribution, colors):
    plt.pie(distribution.values(), labels=distribution.keys(), autopct='%1.1f%%', colors=colors, startangle=140)
    plt.axis('equal')

def bar_chart(distribution, colors):
    plt.bar(distribution.keys(), distribution.values(), color=colors)

def main(args):
    check_args(args)
    distribution = get_distribution(args.dataset)
    colors = plt.cm.viridis(np.linspace(0, 1, len(distribution)))
    
    plt.figure()

    plt.subplot(1, 2, 1)
    pie_chart(distribution, colors)

    plt.subplot(1, 2, 2)
    bar_chart(distribution, colors)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distribution analysis')
    parser.add_argument('dataset', type=str, nargs='?', default='../datasets/images/Apple', help='path to dataset')

    args = parser.parse_args()

    ascii_art = f.renderText('Distribution analysis')
    print(ascii_art)

    main(args)