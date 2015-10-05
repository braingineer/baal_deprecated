#!/usr/bin/env python3
"""
Author: Christopher Perrin

This software converts dependency structures to phrase structures. It follows a algorithm by Xia Fei and Martha Palmer.
"""

from nltk import parse
import argparse
import json
import converter
import os


def main():
    parser = argparse.ArgumentParser(description="Converts a dependency structure to a phrase structure")
    parser.add_argument("data", type=str, help="Dependency graph in Malt-TAB format")
    parser.add_argument("projections", type=argparse.FileType(mode="r"), help="Projections in JSON format")
    parser.add_argument("arguments", type=argparse.FileType(mode="r"), help="Argument table in JSON format")
    parser.add_argument("modifiers", type=argparse.FileType(mode="r"), help="Modification table in JSON format")
    parser.add_argument("-d", "--draw", action='store_true', help="Draw the graph (windowmanager needed)")
    parser.add_argument("-q", "--qtree", action='store_true', help="Print latex qTree to the console")
    parser.add_argument("-o", "--outfile", dest="outfile", type=argparse.FileType(mode="w"), help="Write tree to file")
    args = parser.parse_args()

    deps = parse.DependencyGraph.load(args.data)
    proj = json.load(args.projections)
    arg = json.load(args.arguments)
    mod = json.load(args.modifiers)

    for dep in deps:
        c = converter.Converter(dep, proj, arg, mod)
        ret = c.convert()

        print(ret.pprint())
        if args.qtree:
            print(ret.pprint_latex_qtree())
        if args.outfile:
            args.outfile.write(ret.pprint())
            args.outfile.write(os.linesep)
            args.outfile.flush()
        if args.draw:
            ret.draw()
if __name__ == '__main__':
    main()
